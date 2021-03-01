#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures import logging
from rlstructures.env_wrappers import GymEnv, GymEnvInf
from rlstructures.tools import weight_init
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from rlalgos.a2c_gae.agent import RecurrentAgent, ActionModel
import gym
from gym.wrappers import TimeLimit
from rlalgos.ppo.discrete_ppo import PPO
from rlalgos.a2c_gae.agent import ActionModel, CriticModel, Model


def create_gym_env(env_name):
    return gym.make(env_name)


def create_env(n_envs, env_name=None, max_episode_steps=None, seed=None):
    envs = []
    for k in range(n_envs):
        e = create_gym_env(env_name)
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnv(envs, seed)


def create_train_env(n_envs, env_name=None, max_episode_steps=None, seed=None):
    envs = []
    for k in range(n_envs):
        e = create_gym_env(env_name)
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnvInf(envs, seed)


def create_agent(model, n_actions=1):
    return RecurrentAgent(model=model, n_actions=n_actions)


class Experiment(PPO):
    def __init__(self, config, create_train_env, create_env, create_agent):
        super().__init__(config, create_train_env, create_env, create_agent)

    def _create_model(self):
        action_model = ActionModel(
            self.obs_dim, self.n_actions, self.config["model/hidden_size"]
        )
        critic_model = CriticModel(self.obs_dim, self.config["model/hidden_size"])
        module = Model(action_model, critic_model)
        module.apply(weight_init)
        return module


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    config = {
        "env_name": "CartPole-v0",
        "n_envs": 4,
        "max_episode_steps": 100,
        "discount_factor": 0.9,
        "logdir": "./results",
        "lr": 0.001,
        "n_processes": 4,
        "n_evaluation_processes": 4,
        "n_evaluation_envs": 64,
        "time_limit": 360,
        "coef_critic": 1.0,
        "coef_entropy": 0.01,
        "coef_ppo": 1.0,
        "env_seed": 42,
        "ppo_timesteps": 20,
        "k_epochs": 4,
        "eps_clip": 0.2,
        "gae_coef": 0.3,
        "clip_grad": 2,
        "learner_device": "cpu",
        "evaluation_mode": "stochastic",
        "verbose": True,
        "model/hidden_size": 16,
    }
    exp = Experiment(config, create_train_env, create_env, create_agent)
    exp.run()
