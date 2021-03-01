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
from rlalgos.sac.agent import SACAgent, SACPolicy, SACQ
import gym
from gym.wrappers import TimeLimit
from rlalgos.sac.sac import SAC
from rlalgos.sac.continuouscartopole import ContinuousCartPoleEnv


def create_gym_env(env_name):
    assert env_name == "ContinousCartPole"
    return ContinuousCartPoleEnv()


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


def create_agent(policy, action_dim=None):
    return SACAgent(policy=policy, action_dim=action_dim)


class Experiment(SAC):
    def __init__(self, config, create_train_env, create_env, create_agent):
        super().__init__(config, create_train_env, create_env, create_agent)

    def _create_model(self):
        module = SACPolicy(self.obs_dim, self.action_dim, 16)
        module.apply(weight_init)
        return module

    def _create_q(self):
        module = SACQ(self.obs_dim, self.action_dim, 16)
        module.apply(weight_init)
        return module


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    config = {
        "env_name": "ContinousCartPole",
        "n_envs": 4,
        "n_processes": 4,
        "n_starting_transitions": 1600,
        "batch_timesteps": 1,
        "n_batches_per_epochs": 1,
        "size_batches": 1024,
        "max_episode_steps": 100,
        "tau": 0.005,
        "discount_factor": 0.95,
        "logdir": "./results",
        "replay_buffer_size": 1000000,
        "lr": 0.0003,
        "lambda_entropy": 0.01,
        "n_evaluation_processes": 4,
        "n_evaluation_envs": 64,
        "time_limit": 600,
        "env_seed": 42,
        "clip_grad": 40,
        "learner_device": "cpu",
        "evaluation_mode": "stochastic",
        "verbose": True,
    }
    exp = Experiment(config, create_train_env, create_env, create_agent)
    exp.run()
