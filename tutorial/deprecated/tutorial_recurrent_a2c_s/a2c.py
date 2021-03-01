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
from rlstructures.s_batchers import S_EpisodeBatcher, S_Batcher
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from tutorial.tutorial_recurrent_a2c_s.agent import *

from rlstructures.s_agent import replay_agent


class A2C:
    def __init__(self, config, create_train_env, create_env, create_agent):
        self.config = config

        # Creation of the Logger (that saves in tensorboard and CSV)
        self.logger = TFLogger(log_dir=self.config["logdir"], hps=self.config)

        self._create_env = create_env
        self._create_train_env = create_train_env
        self._create_agent = create_agent

        # Creation of one env instance to get the dimensionnality of observations and number of actions
        env = self._create_env(
            self.config["n_envs"], seed=0, env_name=self.config["env_name"]
        )
        self.n_actions = env.action_space.n
        self.obs_dim = env.reset()[0]["frame"].size()[1]
        del env

    def run(self):
        # Instantiate the learning model abd the baseline model
        action_model = ActionModel(self.obs_dim, self.n_actions, 32)
        critic_model = CriticModel(self.obs_dim, 32)
        self.learning_model = Model(action_model, critic_model)
        self.agent = self._create_agent(
            n_actions=self.n_actions, model=self.learning_model
        )

        # We create a batcher dedicated to evaluation
        model = copy.deepcopy(self.learning_model)
        self.evaluation_batcher = S_EpisodeBatcher(
            n_timesteps=self.config["max_episode_steps"],
            n_slots=self.config["n_evaluation_episodes"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name": self.config["env_name"],
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_threads=self.config["n_evaluation_threads"],
            seeds=[
                self.config["env_seed"] + k * 10
                for k in range(self.config["n_evaluation_threads"])
            ],
            agent_info=DictTensor(
                {"stochastic": torch.tensor([True]).repeat(self.config["n_envs"])}
            ),
            env_info=DictTensor({}),
        )

        # Creation of the batcher for sampling complete pieces of trajectories (i.e Batcher)
        # The batcher will sample n_threads*n_envs trajectories at each call
        # To have a fast batcher, we have to configure it with n_timesteps=self.config["max_episode_steps"]
        model = copy.deepcopy(self.learning_model)
        self.train_batcher = S_Batcher(
            n_timesteps=self.config["a2c_timesteps"],
            n_slots=self.config["n_envs"] * self.config["n_threads"],
            create_agent=self._create_agent,
            create_env=self._create_train_env,
            env_args={
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name": self.config["env_name"],
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_threads=self.config["n_threads"],
            seeds=[
                self.config["env_seed"] + k * 10
                for k in range(self.config["n_threads"])
            ],
            agent_info=DictTensor(
                {"stochastic": torch.tensor([True]).repeat(self.config["n_envs"])}
            ),
            env_info=DictTensor({}),
        )

        # Creation of the optimizer
        optimizer = torch.optim.Adam(
            self.learning_model.parameters(), lr=self.config["lr"]
        )

        # Training Loop:
        _start_time = time.time()
        self.iteration = 0

        # #We launch the evaluation batcher (in deterministic mode)
        n_episodes = self.config["n_evaluation_episodes"]
        agent_info = DictTensor(
            {"stochastic": torch.tensor([False]).repeat(n_episodes)}
        )
        self.evaluation_batcher.execute(n_episodes=n_episodes, agent_info=agent_info)
        self.evaluation_iteration = self.iteration

        # Initialize the training batcher such that agents will start to acqire pieces of episodes
        self.train_batcher.update(self.learning_model.state_dict())
        n_episodes = self.config["n_envs"] * self.config["n_threads"]
        agent_info = DictTensor({"stochastic": torch.tensor([True]).repeat(n_episodes)})
        self.train_batcher.reset(agent_info=agent_info)

        while time.time() - _start_time < self.config["time_limit"]:
            # Call the batcher to get a sample of trajectories

            # 2) We get the pieces of episodes
            self.train_batcher.execute()
            trajectories, info = self.train_batcher.get(blocking=True)
            # if trajectories is None: #All the agents have finished their jobs on the previous episodes:
            #     #Then, reset  again to start new episodes
            #     n_episodes=self.config["n_envs"]*self.config["n_threads"]
            #     agent_info=DictTensor({"stochastic":torch.tensor([True]).repeat(n_episodes)})
            #     self.train_batcher.reset(agent_info=agent_info)
            #     self.train_batcher.execute()
            #     trajectories,info=self.train_batcher.get(blocking=True)

            # 3) Now, we compute the loss
            dt = self.get_loss(trajectories, info)
            [self.logger.add_scalar(k, dt[k].item(), self.iteration) for k in dt.keys()]
            # Computation of final loss
            ld = self.config["critic_coef"] * dt["critic_loss"]
            lr = self.config["a2c_coef"] * dt["a2c_loss"]
            le = self.config["entropy_coef"] * dt["entropy_loss"]

            floss = ld - le - lr
            floss = floss / n_episodes * trajectories.n_elems()

            optimizer.zero_grad()
            floss.backward()

            if self.config["clip_grad"] > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    self.learning_model.parameters(), self.config["clip_grad"]
                )
                self.logger.add_scalar("grad_norm", n.item(), self.iteration)

            optimizer.step()

            # Update the train batcher with the updated model
            self.train_batcher.update(self.learning_model.state_dict())
            self.iteration += 1

            # We check the evaluation batcher
            evaluation_trajectories, info = self.evaluation_batcher.get(blocking=False)
            if not evaluation_trajectories is None:  # trajectories are available
                # Compute the cumulated reward
                cumulated_reward = (
                    (
                        evaluation_trajectories["_observation/reward"]
                        * evaluation_trajectories.mask()
                    )
                    .sum(1)
                    .mean()
                )
                self.logger.add_scalar(
                    "evaluation_reward",
                    cumulated_reward.item(),
                    self.evaluation_iteration,
                )
                print(
                    "At iteration %d, reward is %f"
                    % (self.evaluation_iteration, cumulated_reward.item())
                )
                # We reexecute the evaluation batcher (with same value of agent_info and same number of episodes)
                self.evaluation_batcher.update(self.learning_model.state_dict())
                self.evaluation_iteration = self.iteration
                self.evaluation_batcher.reexecute()

        self.train_batcher.close()
        self.evaluation_batcher.get()  # To wait for the last trajectories
        self.evaluation_batcher.close()
        self.logger.update_csv()  # To save as a CSV file in logdir
        self.logger.close()

    def get_loss(self, trajectories, info):
        # First, we want to compute the cumulated reward per trajectory
        # The reward is a t+1 in each iteration (since it is btained after the aaction), so we use the '_reward' field in the trajectory
        # The 'reward' field corresopnds to the reward at time t
        reward = trajectories["_observation/reward"]

        # We get the mask that tells which transition is in a trajectory (1) or not (0)
        mask = trajectories.mask()

        # We remove the reward values that are not in the trajectories
        reward = reward * mask
        max_length = trajectories.lengths.max().item()

        replayed = replay_agent(self.agent, trajectories, info)
        # Now, we want to compute the action probabilities over the trajectories such that we will be able to do 'backward'
        action_probabilities = replayed["action_probabilities"]
        # We compute the temporal difference
        target = (
            reward
            + self.config["discount_factor"]
            * (1 - trajectories["_observation/done"].float())
            * replayed["_critic"].squeeze(-1).detach()
        )
        td = replayed["critic"].squeeze(-1) - target

        critic_loss = td ** 2
        avg_critic_loss = critic_loss.mean()

        action_distribution = torch.distributions.Categorical(action_probabilities)
        log_proba = action_distribution.log_prob(trajectories["action/action"])
        a2c_loss = -log_proba * td.detach()
        avg_a2c_loss = a2c_loss.mean()

        entropy = action_distribution.entropy()
        avg_entropy = entropy.mean()

        return DictTensor(
            {
                "critic_loss": avg_critic_loss,
                "a2c_loss": avg_a2c_loss,
                "entropy_loss": avg_entropy,
            }
        )
