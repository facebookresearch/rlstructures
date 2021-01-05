from rlstructures import logging
from rlstructures.env_wrappers import GymEnv
from rlstructures.tools import weight_init
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
import gym
from gym.wrappers import TimeLimit
from rlalgos.a2c.a2c_episodes import A2CGAE
from rlalgos.a2c.agent import NNAgent, GRUAgentModel

import gym.spaces as spaces
from gym import ObservationWrapper

import hydra
from omegaconf import DictConfig, OmegaConf


class MyWrapper(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super(MyWrapper, self).__init__(env)
        self.observation_space = None #spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return [observation[0],observation[2]]

def create_gym_env(args):
    return gym.make(args["environment/env_name"])

def create_env(n_envs, max_episode_steps=None, seed=None,**args):
    envs=[]
    for k in range(n_envs):
        e = create_gym_env(args)
        e = MyWrapper(e)
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnv(envs, seed)
    
def create_agent(model,n_actions=1):
    return NNAgent(model=model, n_actions=n_actions)

class Experiment(A2CGAE):
    def __init__(self, config, create_env, create_agent):
        super().__init__(config, create_env, create_agent)

    def _create_model(self):
        module = GRUAgentModel(self.obs_dim, self.n_actions,self.config["model/hidden_size"])
        module.apply(weight_init)
        return module

def flatten(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)         

@hydra.main()
def my_app(cfg : DictConfig) -> None:
    f=flatten(cfg)
    print(f)
    exp = Experiment(f, create_env, create_agent)
    exp.go()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    my_app()
