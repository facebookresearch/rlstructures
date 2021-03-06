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
from rlalgos.a2c_gae.agent import RecurrentAgent, Model, ActionModel, CriticModel
from rlalgos.a2c_gae.a2c import A2C
import gym
from gym.wrappers import TimeLimit
from gym import ObservationWrapper


class MyWrapper(ObservationWrapper):
    """Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(MyWrapper, self).__init__(env)
        self.observation_space = None  # spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return [observation[0], observation[2]]


# We write the 'create_env' and 'create_agent' function in the main file to allow these functions to be used with pickle when creating the batcher processes
def create_gym_env(env_name):
    return gym.make(env_name)


def create_env(n_envs, env_name=None, max_episode_steps=None, seed=None):
    envs = []
    for k in range(n_envs):
        e = create_gym_env(env_name)
        # e = MyWrapper(e)
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnv(envs, seed)


def create_train_env(n_envs, env_name=None, max_episode_steps=None, seed=None):
    envs = []
    for k in range(n_envs):
        e = create_gym_env(env_name)
        # e = MyWrapper(e)
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnvInf(envs, seed)


def create_agent(model, n_actions=1):
    return RecurrentAgent(model=model, n_actions=n_actions)


class Experiment(A2C):
    def __init__(self, config, create_train_env, create_env, create_agent):
        super().__init__(config, create_train_env, create_env, create_agent)

    def _create_model(self):
        am = ActionModel(self.obs_shape[1], self.n_actions, self.config["hidden_size"])
        cm = CriticModel(self.obs_shape[1], self.config["hidden_size"])
        model = Model(am, cm)
        model.apply(weight_init)
        return model


if __name__ == "__main__":
    # We use spawn mode such that most of the environment will run in multiple processes
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    config = {
        "env_name": "CartPole-v0",
        "a2c_timesteps": 20,
        "n_envs": 4,
        "max_episode_steps": 100,
        "env_seed": 42,
        "n_processes": 4,
        "n_evaluation_processes": 2,
        "n_evaluation_envs": 128,
        "time_limit": 3600,
        "lr": 0.001,
        "hidden_size": 32,
        "discount_factor": 0.9,
        "critic_coef": 1.0,
        "entropy_coef": 0.01,
        "a2c_coef": 0.1,
        "gae_coef": 0.3,
        "logdir": "./results",
        "clip_grad": 40,
        "learner_device": "cpu",
        "save_every": 100,
        "optim":"Adam"
    }
    exp = Experiment(config, create_train_env, create_env, create_agent)
    exp.run()
