#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlalgos.logger import Logger, TFLogger
from rlstructures import DictTensor, TemporalDictTensor, Trajectories
from rlalgos.tools import weight_init
from rlstructures.rl_batchers import RL_Batcher
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from rlalgos.reinforce.agent import *
from rlstructures import replay_agent

class Reinforce:
    def __init__(self, config, create_env,  create_agent):
        self.config = config

        self.logger = TFLogger(log_dir=self.config["logdir"], hps=self.config)

        self._create_env = create_env
        self._create_agent = create_agent

    def _state_dict(self, model, device="cpu"):
        sd = model.state_dict()
        for k, v in sd.items():
            sd[k] = v.to(device)
        return sd

    def run(self):
        # Creation of one env instance to get the dimensionnality of observations and number of actions
        env = self._create_env(
            self.config["n_envs"], seed=0, env_name=self.config["env_name"],device=self.config["learner_device"]
        )
        self.n_actions = env.action_space.n
        self.obs_dim = env.reset()[0]["frame"].size()[1]
        del env

        self.learning_model = self._create_model()

        model = self.learning_model
        self.evaluation_batcher = RL_Batcher(
            n_timesteps=self.config["max_episode_steps"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "n_envs": self.config["n_evaluation_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name": self.config["env_name"],
                "device": self.config["evaluation_device"]
            },
            agent_args={"n_actions": self.n_actions, "model": model,"device":self.config["evaluation_device"]},
            n_processes=self.config["n_evaluation_processes"],
            seeds=[
                self.config["env_seed"] + k * 10
                for k in range(self.config["n_evaluation_processes"])
            ],
            agent_info=DictTensor({"stochastic": torch.tensor([True])}),
            env_info=DictTensor({}),
            device=self.config["evaluation_device"]
        )

        model = self.learning_model
        self.train_batcher = RL_Batcher(
            n_timesteps=self.config["max_episode_steps"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name": self.config["env_name"],
                "device":self.config["batcher_device"]
            },
            agent_args={"n_actions": self.n_actions, "model": model,"device":self.config["batcher_device"]},
            n_processes=self.config["n_processes"],
            seeds=[
                self.config["env_seed"] + k * 10
                for k in range(self.config["n_processes"])
            ],
            agent_info=DictTensor({"stochastic": torch.tensor([True])}),
            env_info=DictTensor({}),
            device=self.config["batcher_device"]
        )

        self.learning_model.to(self.config["learner_device"])
        self.agent = self._create_agent(
            n_actions=self.n_actions, model=self.learning_model,device=self.config["learner_device"],copy_model=False
        )
        optimizer = torch.optim.RMSprop(
            self.learning_model.parameters(), lr=self.config["lr"]
        )

        _start_time = time.time()
        self.iteration = 0

        # We launch the evaluation batcher, such that it starts to collect trajectories with the current model
        n_episodes = (
            self.config["n_evaluation_processes"] * self.config["n_evaluation_envs"]
        )
        agent_info = DictTensor(
            {
                "stochastic": torch.tensor(
                    [self.config["evaluation_mode"] == "stochastic"]
                ).repeat(n_episodes)
            }
        )
        self.evaluation_batcher.reset(agent_info=agent_info)
        self.evaluation_batcher.execute()
        self.evaluation_iteration = self.iteration

        # Update the batcher with the last version of the learning model
        sd=self._state_dict(self.learning_model,torch.device("cpu"))
        self.train_batcher.update(sd)

        n_interactions = 0

        while time.time() - _start_time < self.config["time_limit"]:
            # 1) The policy will be executed in "stochastic' mode
            n_episodes = self.config["n_envs"] * self.config["n_processes"]
            agent_info = DictTensor(
                {"stochastic": torch.tensor([True]).repeat(n_episodes)}
            )
            self.train_batcher.reset(agent_info=agent_info)
            self.train_batcher.execute()

            # 2) We get the trajectories (and wait until the trajectories have been sampled)
            trajectories, n_env_running = self.train_batcher.get(blocking=True)

            assert n_env_running == 0  # Assert that all trajectories are finished
            n_interactions += trajectories.trajectories.mask().sum().item()
            self.logger.add_scalar(
                "n_interactions_per_seconds",
                n_interactions / (time.time() - _start_time),
                self.iteration,
            )

            # 3) Compute the loss
            dt = self.get_loss(trajectories)
            [
                self.logger.add_scalar("loss/" + k, dt[k].item(), self.iteration)
                for k in dt.keys()
            ]

            # 4) Compute the final loss by linear combination of the different individual losses
            ld = self.config["baseline_coef"] * dt["baseline_loss"]
            lr = self.config["reinforce_coef"] * dt["reinforce_loss"]
            le = self.config["entropy_coef"] * dt["entropy_loss"]
            floss = ld - le - lr

            # 5) Update the parameters of the model
            optimizer.zero_grad()
            floss.backward()
            optimizer.step()

            # 6) Update the train batcher with the updated model
            sd=self._state_dict(self.learning_model,torch.device("cpu"))
            self.train_batcher.update(sd)

            # 7) Print some messages
            print(
                "At iteration %d, avg (discounted) reward is %f"
                % (self.iteration, dt["avg_reward"].item())
            )
            print(
                "\t Avg trajectory length is %f"
                % (trajectories.trajectories.lengths.float().mean().item())
            )
            print(
                "\t Curves can be visualized using 'tensorboard --logdir=%s'"
                % self.config["logdir"]
            )
            self.iteration += 1

            #8)---- Evaluation
            evaluation_trajectories, n_env_running = self.evaluation_batcher.get(
                blocking=False
            )
            if not evaluation_trajectories is None:  # trajectories are available
                assert n_env_running == 0
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
                    "evaluation_reward/" + self.config["evaluation_mode"],
                    cumulated_reward.item(),
                    self.evaluation_iteration,
                )
                print(
                    "-- Iteration ",
                    self.iteration,
                    " Evaluation reward = ",
                    cumulated_reward.item(),
                )
                # We reexecute the evaluation batcher to start the acquisition of new trajectories
                sd=self._state_dict(self.learning_model,torch.device("cpu"))
                self.evaluation_batcher.update(sd)
                self.evaluation_iteration = self.iteration
                n_episodes = (
                    self.config["n_evaluation_processes"]
                    * self.config["n_evaluation_envs"]
                )
                agent_info = DictTensor(
                    {
                        "stochastic": torch.tensor(
                            [self.config["evaluation_mode"] == "stochastic"]
                        ).repeat(n_episodes)
                    }
                )
                self.evaluation_batcher.reset(agent_info=agent_info)
                self.evaluation_batcher.execute()

        self.train_batcher.close()
        self.evaluation_batcher.get()  # To wait for the last trajectories
        self.evaluation_batcher.close()
        self.logger.update_csv()  # To save as a CSV file in logdir
        self.logger.close()

    def get_loss(self, trajectories):
        trajectories=trajectories.to(self.config["learner_device"])
        replayed = replay_agent(self.agent, trajectories)

        info = trajectories.info
        trajectories = trajectories.trajectories

        # Compute the cumulated future reward
        reward = trajectories["_observation/reward"]
        mask = trajectories.mask()
        reward = reward * mask
        max_length = trajectories.lengths.max().item()
        cumulated_reward = torch.zeros_like(reward).to(reward.device)
        cumulated_reward[:, max_length - 1] = reward[:, max_length - 1]
        for t in range(max_length - 2, -1, -1):
            cumulated_reward[:, t] = (
                reward[:, t]
                + self.config["discount_factor"] * cumulated_reward[:, t + 1]
            )

        # Compute reinforce loss
        action_probabilities = replayed["action_probabilities"]
        action_distribution = torch.distributions.Categorical(action_probabilities)
        baseline = replayed["baseline"].squeeze(-1)
        log_proba = action_distribution.log_prob(trajectories["action/action"])
        reinforce_loss = log_proba * (cumulated_reward - baseline).detach()
        reinforce_loss = (reinforce_loss * mask).sum(1) / mask.sum(1)
        avg_reinforce_loss = reinforce_loss.mean()

        # Compute entropy loss
        entropy = action_distribution.entropy()
        entropy = (entropy * mask).sum(1) / mask.sum(1)
        avg_entropy = entropy.mean()

        # Compute baseline loss
        baseline_loss = (baseline - cumulated_reward) ** 2
        baseline_loss = (baseline_loss * mask).sum(1) / mask.sum(1)
        avg_baseline_loss = baseline_loss.mean()

        return DictTensor(
            {
                "avg_reward": cumulated_reward[:, 0].mean(),
                "baseline_loss": avg_baseline_loss,
                "reinforce_loss": avg_reinforce_loss,
                "entropy_loss": avg_entropy,
            }
        )
