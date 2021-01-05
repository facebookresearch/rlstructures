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
from rlalgos.sac.agent import SACAgent,SACPolicy,SACQ
import gym
from gym.wrappers import TimeLimit
from rlalgos.sac.sac import SAC
from rlalgos.envs.continuouscartopole import ContinuousCartPoleEnv

import hydra
from omegaconf import DictConfig, OmegaConf

def create_gym_env(args):
    return ContinuousCartPoleEnv()

def create_env(n_envs, mode="train",max_episode_steps=None, seed=None,**args):
    envs=[]
    for k in range(n_envs):
        e = create_gym_env(args)
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
        
    if mode=="train":
        return GymEnvInf(envs, seed)
    else:
        return GymEnv(envs, seed)

    
def create_agent(policy,action_dim=None):
    return SACAgent(policy=policy, action_dim=action_dim)

class Experiment(SAC):
    def __init__(self, config, create_env, create_agent):
        super().__init__(config, create_env, create_agent)

    def _create_model(self):
        module = SACPolicy(self.obs_dim, self.action_dim, 16)
        module.apply(weight_init)
        return module

    def _create_q(self):
        module = SACQ(self.obs_dim, self.action_dim, 16)
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
