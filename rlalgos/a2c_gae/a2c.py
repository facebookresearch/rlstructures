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
from rlstructures.rl_batchers import RL_Batcher
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from rlstructures import replay_agent
import torch.optim as optim

class A2C:
    def __init__(self, config, create_train_env, create_env, create_agent):
        self.config = config

        # Creation of the Logger (that saves in tensorboard and CSV)
        self.logger = TFLogger(
            log_dir=self.config["logdir"],
            hps=self.config,
            save_every=self.config["save_every"],
        )

        self._create_env = create_env
        self._create_train_env = create_train_env
        self._create_agent = create_agent

        # Creation of one env instance to get the dimensionnality of observations and number of actions
        env = self._create_env(
            self.config["n_envs"], seed=0, env_name=self.config["env_name"]
        )
        self.n_actions = env.action_space.n
        self.obs_shape = env.reset()[0]["frame"].size()
        del env

    def _state_dict(self, model, device="cpu"):
        sd = model.state_dict()
        for k, v in sd.items():
            sd[k] = v.to(device)
        return sd

    def run(self):
        # Instantiate the learning model abd the baseline model
        self.learning_model = self._create_model()

        self.agent = self._create_agent(
            n_actions=self.n_actions, model=self.learning_model
        )

        # We create a batcher dedicated to evaluation
        model = copy.deepcopy(self.learning_model)
        self.evaluation_batcher = RL_Batcher(
            n_timesteps=self.config["max_episode_steps"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "n_envs": self.config["n_evaluation_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name": self.config["env_name"],
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_processes=self.config["n_evaluation_processes"],
            seeds=[
                self.config["env_seed"] + k * 10
                for k in range(self.config["n_evaluation_processes"])
            ],
            agent_info=DictTensor({"stochastic": torch.tensor([True])}),
            env_info=DictTensor({}),
        )

        # Creation of the batcher for sampling complete pieces of trajectories (i.e Batcher)
        # The batcher will sample n_threads*n_envs trajectories at each call
        # To have a fast batcher, we have to configure it with n_timesteps=self.config["max_episode_steps"]
        model = copy.deepcopy(self.learning_model)
        self.train_batcher = RL_Batcher(
            n_timesteps=self.config["a2c_timesteps"],
            create_agent=self._create_agent,
            create_env=self._create_train_env,
            env_args={
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name": self.config["env_name"],
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_processes=self.config["n_processes"],
            seeds=[
                self.config["env_seed"] + k * 10
                for k in range(self.config["n_processes"])
            ],
            agent_info=DictTensor({"stochastic": torch.tensor([True])}),
            env_info=DictTensor({}),
        )

        # Creation of the optimizer
        self.learning_model.to(self.config["learner_device"])
        optimizer = getattr(torch.optim, self.config["optim"])(
            self.learning_model.parameters(), lr=self.config["lr"]
        )

        # Training Loop:
        _start_time = time.time()
        self.iteration = 0

        # #We launch the evaluation batcher (in deterministic mode)
        n_episodes = (
            self.config["n_evaluation_processes"] * self.config["n_evaluation_envs"]
        )
        agent_info = DictTensor(
            {"stochastic": torch.tensor([False]).repeat(n_episodes)}
        )
        self.evaluation_batcher.reset(agent_info=agent_info)
        self.evaluation_batcher.execute()
        self.evaluation_iteration = self.iteration

        # Initialize the training batcher such that agents will start to acqire pieces of episodes
        self.train_batcher.update(self._state_dict(self.learning_model))
        n_episodes = self.config["n_envs"] * self.config["n_processes"]
        agent_info = DictTensor({"stochastic": torch.tensor([True]).repeat(n_episodes)})
        self.train_batcher.reset(agent_info=agent_info)

        while time.time() - _start_time < self.config["time_limit"]:
            self.train_batcher.execute()
            trajectories, n = self.train_batcher.get(blocking=True)
            assert n == self.config["n_envs"] * self.config["n_processes"]
            # print(trajectories.trajectories["_observation/reward"].sum(1))
            dt = self.get_loss(trajectories)

            [
                self.logger.add_scalar("loss/" + k, dt[k].item(), self.iteration)
                for k in dt.keys()
            ]
            ld = self.config["critic_coef"] * dt["critic_loss"]
            lr = self.config["a2c_coef"] * dt["a2c_loss"]
            le = self.config["entropy_coef"] * dt["entropy_loss"]
            floss = ld - le - lr

            optimizer.zero_grad()
            floss.backward()

            if self.config["clip_grad"] > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    self.learning_model.parameters(), self.config["clip_grad"]
                )
                self.logger.add_scalar("grad_norm", n.item(), self.iteration)

            optimizer.step()

            # Update the train batcher with the updated model
            self.train_batcher.update(self._state_dict(self.learning_model))
            self.iteration += 1

            # We check the evaluation batcher
            evaluation_trajectories, n = self.evaluation_batcher.get(blocking=False)
            if not evaluation_trajectories is None:  # trajectories are available
                # Compute the cumulated reward
                cumulated_reward = (
                    (
                        evaluation_trajectories.trajectories["_observation/reward"]
                        * evaluation_trajectories.trajectories.mask()
                    )
                    .sum(1)
                    .mean()
                )
                self.logger.add_scalar(
                    "evaluation_reward",
                    cumulated_reward.item(),
                    self.evaluation_iteration,
                )
                # print("At iteration %d, reward is %f"%(self.evaluation_iteration,cumulated_reward.item()))
                # We reexecute the evaluation batcher (with same value of agent_info and same number of episodes)
                self.evaluation_batcher.update(self._state_dict(self.learning_model))
                self.evaluation_iteration = self.iteration
                n_episodes = (
                    self.config["n_evaluation_processes"]
                    * self.config["n_evaluation_envs"]
                )
                agent_info = DictTensor(
                    {"stochastic": torch.tensor([False]).repeat(n_episodes)}
                )
                self.evaluation_batcher.reset(agent_info=agent_info)
                self.evaluation_batcher.execute()

            if time.time() - _start_time > 600 and self.iteration % 1000 == 0:
                self.logger.update_csv()

        self.train_batcher.close()
        self.evaluation_batcher.get()  # To wait for the last trajectories
        self.evaluation_batcher.close()
        self.logger.update_csv()  # To save as a CSV file in logdir
        self.logger.close()

    def get_loss(self, trajectories):
        trajectories = trajectories.to(self.config["learner_device"])
        print(trajectories.trajectories["agent_state/agent_step"][0])
        print(trajectories.trajectories["_agent_state/agent_step"][0])
        replayed = replay_agent(
            self.agent, trajectories, replay_method_name="call_replay"
        )
        info = trajectories.info
        trajectories = trajectories.trajectories
        # First, we want to compute the cumulated reward per trajectory
        # The reward is a t+1 in each iteration (since it is btained after the aaction), so we use the '_reward' field in the trajectory
        # The 'reward' field corresopnds to the reward at time t
        reward = trajectories["_observation/reward"]

        # Now, we want to compute the action probabilities over the trajectories such that we will be able to do 'backward'
        action_probabilities = replayed["action_probabilities"]
        # We compute the temporal difference
        gae = self.get_gae(
            trajectories["_observation/done"],
            reward,
            replayed["critic"].squeeze(-1),
            replayed["_critic"].squeeze(-1),
            self.config["discount_factor"],
            self.config["gae_coef"],
        )
        td = gae

        critic_loss = td ** 2
        avg_critic_loss = critic_loss.mean()

        action_distribution = torch.distributions.Categorical(action_probabilities)
        log_proba = action_distribution.log_prob(trajectories["action/action"])
        a2c_loss = log_proba * td.detach()
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

    def get_gae(self, done, reward, critic, _critic, discount_factor=1, _lambda=0):
        r = reward
        d = done.float()
        target = r + discount_factor * _critic.detach() * (1.0 - d)

        delta = target - critic
        T = done.size()[1]
        gae = delta[:, -1]
        gaes = [gae]
        for t in range(T - 2, -1, -1):
            gae = delta[:, t] + discount_factor * _lambda * (1 - d[:, t]) * gae
            gaes.append(gae)
        gaes = list([g.unsqueeze(-1) for g in reversed(gaes)])
        fgae = torch.cat(gaes, dim=1)
        return fgae
