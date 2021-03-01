#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from rlstructures.env_wrappers import GymEnv, GymEnvInf
from rlstructures.tools import weight_init
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from rlalgos.a2c_gae.atari_agent import AtariAgent, ActionModel, CriticModel, Model
from rlalgos.a2c_gae.a2c import A2C
import gym
from gym.wrappers import TimeLimit
from gym import ObservationWrapper
from rlalgos.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch


def create_env(n_envs, env_name, max_episode_steps=None, seed=None, **args):
    envs = []
    for k in range(n_envs):
        e = make_atari(env_name)
        e = wrap_deepmind(e)
        e = wrap_pytorch(e)
        envs.append(e)
    return GymEnv(envs, seed)


def create_train_env(n_envs, env_name, max_episode_steps=None, seed=None, **args):
    envs = []
    for k in range(n_envs):
        e = make_atari(env_name)
        e = wrap_deepmind(e)
        e = wrap_pytorch(e)
        envs.append(e)
    return GymEnvInf(envs, seed)


def create_agent(model, n_actions=1):
    return AtariAgent(model=model, n_actions=n_actions)


class Experiment(A2C):
    def __init__(self, config, create_train_env, create_env, create_agent):
        super().__init__(config, create_train_env, create_env, create_agent)

    def _create_model(self):
        am = ActionModel(self.obs_shape, self.n_actions)
        cm = CriticModel(self.obs_shape)
        model = Model(am, cm)
        # model.apply(weight_init)
        return model


if __name__ == "__main__":
    # We use spawn mode such that most of the environment will run in multiple processes
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    config = {
        "env_name": "PongNoFrameskip-v4",
        "a2c_timesteps": 1,
        "n_envs": 4,
        "max_episode_steps": 15000,
        "env_seed": 42,
        "n_processes": 4,
        "n_evaluation_processes": 4,
        "n_evaluation_envs": 1,
        "time_limit": 3600,
        "lr": 0.0001,
        "discount_factor": 0.95,
        "critic_coef": 1.0,
        "entropy_coef": 0.01,
        "a2c_coef": 1.0,
        "gae_coef": 0.3,
        "logdir": "./results",
        "clip_grad": 0,
        "learner_device": "cpu",
        "save_every": 1,
        "optim": "RMSprop",
    }
    exp = Experiment(config, create_train_env, create_env, create_agent)
    exp.run()
