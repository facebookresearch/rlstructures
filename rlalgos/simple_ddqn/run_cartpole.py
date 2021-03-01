#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures import logging
from rlstructures.env_wrappers import GymEnv, GymEnvInf
from rlstructures.tools import weight_init
from rlstructures.batchers import Batcher, EpisodeBatcher
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from rlalgos.simple_ddqn.agent import QAgent, DQMLP
import gym
from gym.wrappers import TimeLimit
from rlalgos.simple_ddqn.ddqn import DQN


def create_gym_env(args):
    return gym.make(args["environment/env_name"])


def create_env(n_envs, mode="train", max_episode_steps=None, seed=None, **args):
    envs = []
    for k in range(n_envs):
        e = create_gym_env(args)
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)

    if mode == "train":
        return GymEnvInf(envs, seed)
    else:
        return GymEnv(envs, seed)


def create_agent(
    n_actions=None,
    model=None,
):
    return QAgent(model=model, n_actions=n_actions)


class Experiment(DQN):
    def __init__(self, config, create_env, create_agent):
        super().__init__(config, create_env, create_agent)

    def _create_model(self):
        module = DQMLP(self.obs_shape[0], self.n_actions, 64)

        module.apply(weight_init)
        return module


def flatten(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    # We use spawn mode such that most of the environment will run in multiple processes
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    config = {
        "environment/env_name": "CartPole-v0",
        "n_envs": 4,
        "max_episode_steps": 10000,
        "discount_factor": 0.99,
        "epsilon_greedy": 0.1,
        "replay_buffer_size": 10000,
        "n_batches": 32,
        "initial_buffer_epochs": 1,
        "qvalue_epochs": 1,
        "batch_timesteps": 4,
        "lr": 0.01,
        "n_processes": 4,
        "n_evaluation_processes": 4,
        "verbose": True,
        "n_evaluation_envs": 32,
        "time_limit": 28800,
        "env_seed": 42,
        "clip_grad": 0.0,
        "learner_device": "cpu",
        "optim": "Adam",
        "update_target_tau": 0.005,
        "logdir": "./results",
        "save_every": 1,
    }
    exp = Experiment(config, create_env, create_agent)
    exp.run()
