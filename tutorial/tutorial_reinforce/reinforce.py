#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures.logger import Logger, TFLogger
from rlstructures import DictTensor, TemporalDictTensor
from rlstructures import logging
from rlstructures.tools import weight_init
from rlstructures.batchers import EpisodeBatcher
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from tutorial.tutorial_reinforce.agent import *

class Reinforce:
    def __init__(self, config,create_env,create_agent):
        self.config = config

        # Creation of the Logger (that saves in tensorboard and CSV)
        self.logger = TFLogger(log_dir=self.config["logdir"], hps=self.config)

        self._create_env=create_env
        self._create_agent=create_agent

        #Creation of one env instance to get the dimensionnality of observations and number of actions
        env = self._create_env(self.config["n_envs"], seed=0,env_name=self.config["env_name"])
        self.n_actions = env.action_space.n
        self.obs_dim = env.reset()[0]["frame"].size()[1]
        del env

    def run(self):
        # Instantiate the learning model abd the baseline model
        self.learning_model=AgentModel(self.obs_dim,self.n_actions,16)
        self.baseline_model=BaselineModel(self.obs_dim,16)

        #Creation of the batcher for sampling complete episodes (i.e Episode Batcher)
        #The batcher will sample n_threads*n_envs trajectories at each call
        # To have a fast batcher, we have to configure it with n_timesteps=self.config["max_episode_steps"]
        model=copy.deepcopy(self.learning_model)
        self.train_batcher=EpisodeBatcher(
            n_timesteps=self.config["max_episode_steps"],
            n_slots=self.config["n_envs"]*self.config["n_threads"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name":self.config["env_name"]
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_threads=self.config["n_threads"],
            seeds=[self.config["env_seed"]+k*10 for k in range(self.config["n_threads"])],
        )

        #Creation of the optimizer
        optimizer = torch.optim.Adam(nn.Sequential(self.learning_model,self.baseline_model).parameters(), lr=self.config["lr"])

        #Training Loop:
        _start_time=time.time()
        self.iteration=0
        while(time.time()-_start_time<self.config["time_limit"]):

            #Update the batcher with the last version of the learning model
            self.train_batcher.update(self.learning_model.state_dict())

            #Call the batcher to get a sample of trajectories
            #1) The policy will be executed in "stochastic' mode
            n_episodes=self.config["n_envs"]*self.config["n_threads"]
            agent_info=DictTensor({"stochastic":torch.tensor([True]).repeat(n_episodes)})
            self.train_batcher.execute(n_episodes=n_episodes,agent_info=agent_info)

            #2) We get the trajectories (and wait until the trajectories have been sampled)
            trajectories=self.train_batcher.get(blocking=True)

            #3) Now, we compute the loss
            dt=self.get_loss(trajectories)
            [self.logger.add_scalar(k,dt[k].item(),self.iteration) for k in dt.keys()]

            # Computation of final loss
            ld = self.config["baseline_coef"] * dt["baseline_loss"]
            lr = self.config["reinforce_coef"] * dt["reinforce_loss"]
            le = self.config["entropy_coef"] * dt["entropy_loss"]

            floss = ld - le - lr

            optimizer.zero_grad()
            floss.backward()
            optimizer.step()

            #Update the train batcher with the updated model
            self.train_batcher.update(self.learning_model.state_dict())
            print("At iteration %d, avg (discounted) reward is %f"%(self.iteration,dt["avg_reward"].item()))
            print("\t Avg trajectory length is %f"%(trajectories.lengths.float().mean().item()))
            print("\t Curves can be visualized using 'tensorbaord --logdir=%s'"%self.config["logdir"])
            self.iteration+=1

        self.train_batcher.close()
        self.logger.update_csv() # To save as a CSV file in logdir
        self.logger.close()

    def get_loss(self,trajectories):
            #First, we want to compute the cumulated reward per trajectory
            #The reward is a t+1 in each iteration (since it is btained after the aaction), so we use the '_reward' field in the trajectory
            # The 'reward' field corresopnds to the reward at time t
            reward=trajectories["_reward"]

            #We get the mask that tells which transition is in a trajectory (1) or not (0)
            mask=trajectories.mask()

            #We remove the reward values that are not in the trajectories
            reward=reward*mask

            #We compute the future cumulated reward at each timestep (by reverse computation)
            max_length=trajectories.lengths.max().item()
            cumulated_reward=torch.zeros_like(reward)
            cumulated_reward[:,max_length-1]=reward[:,max_length-1]
            for t in range(max_length-2,-1,-1):
                cumulated_reward[:,t]=reward[:,t]+self.config["discount_factor"]*cumulated_reward[:,t+1]

            #Now, we want to compute the action probabilities over the trajectories such that we will be able to do 'backward'
            action_probabilities=[]
            for t in range(max_length):
                proba=self.learning_model(trajectories["frame"][:,t])
                action_probabilities.append(proba.unsqueeze(1)) # We append the probability, and introduces the temporal dimension (2nde dimension)
            action_probabilities=torch.cat(action_probabilities,dim=1) #Now, we have a B x T x n_actions tensor

            #We compute the baseline
            baseline=[]
            for t in range(max_length):
                b=self.baseline_model(trajectories["frame"][:,t])
                baseline.append(b.unsqueeze(1))
            baseline=torch.cat(baseline,dim=1).squeeze(-1) #Now, we have a B x T tensor

            #We compute the baseline loss
            baseline_loss=(baseline-cumulated_reward)**2
            #We sum the loss for each episode (considering the mask)
            baseline_loss= (baseline_loss*mask).sum(1)/mask.sum(1)
            #We average the loss over all the trajectories
            avg_baseline_loss = baseline_loss.mean()

            #We do the same on the reinforce loss
            action_distribution=torch.distributions.Categorical(action_probabilities)
            log_proba=action_distribution.log_prob(trajectories["action"])
            reinforce_loss = log_proba * (cumulated_reward-baseline).detach()
            reinforce_loss = (reinforce_loss*mask).sum(1)/mask.sum(1)
            avg_reinforce_loss=reinforce_loss.mean()

            #We compute the entropy loss
            entropy=action_distribution.entropy()
            entropy=(entropy*mask).sum(1)/mask.sum(1)
            avg_entropy=entropy.mean()

            return DictTensor({"avg_reward":cumulated_reward[:,0].mean(),"baseline_loss":avg_baseline_loss,"reinforce_loss":avg_reinforce_loss,"entropy_loss":avg_entropy})
