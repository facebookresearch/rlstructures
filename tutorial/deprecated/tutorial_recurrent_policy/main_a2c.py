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
from tutorial.tutorial_recurrent_policy.agent import RecurrentAgent
from tutorial.tutorial_recurrent_policy.a2c import A2C
import gym
from gym.wrappers import TimeLimit

# We write the 'create_env' and 'create_agent' function in the main file to allow these functions to be used with pickle when creating the batcher processes
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


class Experiment(A2C):
    def __init__(self, config, create_env, create_train_env, create_agent):
        super().__init__(config, create_env, create_train_env, create_agent)


if __name__ == "__main__":
    # We use spawn mode such that most of the environment will run in multiple processes
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    config = {
        "env_name": "CartPole-v0",
        "a2c_timesteps": 3,
        "n_envs": 4,
        "max_episode_steps": 100,
        "env_seed": 42,
        "n_threads": 4,
        "n_evaluation_threads": 2,
        "n_evaluation_episodes": 256,
        "time_limit": 3600,
        "lr": 0.001,
        "discount_factor": 0.95,
        "critic_coef": 1.0,
        "entropy_coef": 0.01,
        "a2c_coef": 1.0,
        "logdir": "./results",
    }
    exp = Experiment(config, create_env, create_train_env, create_agent)
    exp.run()
