#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from rlalgos import BaseExperiment
from rlstructures.logger import Logger, TFLogger
from rlstructures import DictTensor, TemporalDictTensor
from rlstructures.batchers import Batcher,EpisodeBatcher
from rlstructures import logging
from rlstructures.tools import weight_init
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
import pickle

class PPO(BaseExperiment):
    def __init__(self, config, create_env, create_agent):        
        super().__init__(config,create_env,create_agent)
        env = self._create_env(
            self.config["n_envs"], seed=0,**{k:self.config[k] for k in self.config if k.startswith("environment/")}
        )
        self.n_actions = env.action_space.n
        self.obs_dim = env.reset()[0]["frame"].size()[1]        
        del env
    
    def check_arguments(self,args):
        assert args["n_evaluation_rollouts"]%(args["n_envs"]*args["n_evaluation_threads"])==0
        assert args["evaluation_mode"]=="deterministic" or args["evaluation_mode"]=="stochastic"
        return True


    def reset(self):
        #Creation of the batchers
        model=copy.deepcopy(self.learning_model)
        print(self.config)
        self.train_batcher=Batcher(
            n_timesteps=self.config["learning_timesteps"],
            n_slots=self.config["n_envs"]*self.config["n_threads"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "mode":"train",
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                **{k:self.config[k] for k in self.config if k.startswith("environment/")}
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_threads=self.config["n_threads"],
            seeds=self.config["env_seed"],           
        )
        
        model=copy.deepcopy(self.learning_model)
        self.evaluation_batcher=EpisodeBatcher( 
            n_timesteps=self.config["max_episode_steps"],
            n_slots=self.config["n_evaluation_rollouts"],   
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={        
                "mode":"evaluation",                      
                "max_episode_steps": self.config["max_episode_steps"],
                "n_envs": self.config["n_envs"],
                **{k:self.config[k] for k in self.config if k.startswith("environment/")}

            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_threads=self.config["n_evaluation_threads"],
            seeds=self.config["env_seed"]*10,
        )

        self.register_batcher(self.train_batcher)
        self.register_batcher(self.evaluation_batcher)
        

    def _state_dict(self,model,device):
        sd = model.state_dict()
        for k, v in sd.items():
            sd[k] = v.to(device)
        return sd

    def run(self):
        device = torch.device(self.config["learner_device"])
        self.learning_model.to(device)
        optimizer = torch.optim.Adam(
            self.learning_model.parameters(), lr=self.config["lr"]
        )
        cpu_parameters=self._state_dict(self.learning_model,torch.device("cpu"))
        self.train_batcher.update(cpu_parameters)
        self.evaluation_batcher.update(cpu_parameters)
        n_episodes=self.config["n_evaluation_rollouts"]
        self.evaluation_batcher.execute(agent_info=DictTensor({"stochastic":torch.ones(n_episodes)}), n_episodes=n_episodes)

        # Initialize the train batcher
        n_episodes=self.config["n_envs"]*self.config["n_threads"]
        self.train_batcher.reset(agent_info=DictTensor({"stochastic":torch.ones(n_episodes)}))
        
        _start_time=time.time()
        while time.time()-_start_time<self.config["time_limit"]:            
            self.train_batcher.execute()
            trajectories=self.train_batcher.get()
            avg_reward = 0
            for K in range(self.config["k_epochs"]):
                optimizer.zero_grad()
                dt = self.get_loss(trajectories)
                [
                    self.logger.add_scalar("loss/" + k, dt[k].item(), self.iteration)
                    for k in dt.keys()
                ]

                # Computation of final loss
                ld = self.config["coef_critic"] * dt["value_loss"]
                lr = self.config["coef_ppo"] * dt["ppo_loss"]
                le = self.config["coef_entropy"] * dt["entropy_loss"]

                floss = ld - le - lr
                floss.backward()
                if self.config["clip_grad"] > 0:
                    n = torch.nn.utils.clip_grad_norm_(
                        self.learning_model.parameters(), self.config["clip_grad"]
                    )
                    self.logger.add_scalar("grad_norm", n.item(), self.iteration)
                optimizer.step()
                self.evaluate()
                self.iteration+=1
            cpu_parameters=self._state_dict(self.learning_model,torch.device("cpu"))
            self.train_batcher.update(cpu_parameters)
            self.evaluate()
            self.iteration+=1
            

    def evaluate(self,relaunch=True):
        evaluation_trajectories = self.evaluation_batcher.get(blocking=False)
        if (evaluation_trajectories is None):
            return
        
        avg_reward = (
                    (
                        evaluation_trajectories["_reward"]
                        * evaluation_trajectories.mask()
                    )
                    .sum(1)
                    .mean()
                    .item()
        )
        self.logger.add_scalar("avg_reward/"+self.config["evaluation_mode"], avg_reward, self.iteration)
        if (self.config["verbose"]):
                print("Iteration "+str(self.iteration)+", Reward =  "+str(avg_reward))
        
        if (relaunch):
            cpu_parameters=self._state_dict(self.learning_model,torch.device("cpu"))
            self.evaluation_batcher.update(cpu_parameters)
            self.evaluation_batcher.reexecute()
        return avg_reward


    def get_loss(self, trajectories):     
        device=self.config["learner_device"]   
        trajectories = trajectories.to(device)
        max_length = trajectories.lengths.max().item()
        assert trajectories.lengths.eq(max_length).all()
        actions = trajectories["action"]
        actions_probabilities = trajectories["action_probabilities"]
        reward = trajectories["_reward"]
        frame = trajectories["frame"]
        last_action = trajectories["last_action"]
        done = trajectories["_done"].float()
        # Re compute model on trajectories
        n_action_scores = []
        n_values = []
        hidden_state = trajectories["agent_state"][:, 0]
        for T in range(max_length):
            _as, _v, hidden_state = self.learning_model(
                hidden_state, frame[:, T], last_action[:, T]
            )
            n_action_scores.append(_as.unsqueeze(1))
            n_values.append(_v.unsqueeze(1))
        n_action_scores = torch.cat(n_action_scores, dim=1)

        n_values = torch.cat(
            [*n_values, torch.zeros(trajectories.n_elems(), 1, 1).to(device)], dim=1
        ).squeeze(-1)

        # Compute value function for last state
        _idx = torch.arange(trajectories.n_elems()).to(device)
        _hidden_state = hidden_state.detach() #trajectories["_agent_state"][_idx, trajectories.lengths - 1]
        _frame = trajectories["_frame"][_idx, trajectories.lengths - 1]
        _last_action = trajectories["_last_action"][_idx, trajectories.lengths - 1]
        _, _v, _ = self.learning_model(_hidden_state, _frame, _last_action)
        n_values[_idx, trajectories.lengths] = _v.squeeze(-1)
        
        advantage = self.get_gae(
            trajectories,
            n_values,
            discount_factor=self.config["discount_factor"],
            _lambda=self.config["gae_lambda"],
        )
        
        value_loss = advantage ** 2
        avg_value_loss = value_loss.mean()

        n_action_probabilities = torch.softmax(n_action_scores, dim=2)
        n_action_distribution = torch.distributions.Categorical(n_action_probabilities)
        log_a=torch.distributions.Categorical(actions_probabilities).log_prob(actions)
        log_na=n_action_distribution.log_prob(actions)
        ratios=torch.exp(log_na-log_a)
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios,1-self.config["eps_clip"],1-self.config["eps_clip"])*advantage
        
        ppo_loss = torch.min(surr1,surr2)
        avg_ppo_loss = ppo_loss.mean()

        entropy_loss = n_action_distribution.entropy()
        avg_entropy_loss = entropy_loss.mean()

        
        dt = DictTensor(
            {
                "entropy_loss": avg_entropy_loss,
                "ppo_loss": avg_ppo_loss,
                "value_loss": avg_value_loss,
            }
        )
        return dt

    def get_gae(self, trajectories, values, discount_factor=1, _lambda=0):
        r = trajectories["_reward"]
        v = values[:, 1:].detach()
        d = trajectories["_done"].float()
        delta = r + discount_factor * v * (1.0 - d) - values[:, :-1]
        T = trajectories.lengths.max().item()
        gae = delta[:, -1]
        gaes = [gae]
        for t in range(T - 2, -1, -1):
            gae = delta[:, t] + discount_factor * _lambda * (1 - d[:, t]) * gae
            gaes.append(gae)
        gaes = list([g.unsqueeze(-1) for g in reversed(gaes)])
        fgae = torch.cat(gaes, dim=1)
        return fgae