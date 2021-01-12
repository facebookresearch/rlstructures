#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures import logging
from rlstructures.env_wrappers import GymEnv, GymEnvInf
from rlstructures.tools import weight_init
from rlstructures.batchers import Batcher,EpisodeBatcher
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from rlalgos.dqn.agent import QAgent, QMLP, DQMLP
import gym
from gym.wrappers import TimeLimit
from rlalgos.dqn.duelling_dqn import DQN

import hydra
from omegaconf import DictConfig, OmegaConf

def create_gym_env(args):
    return gym.make(args["environment/env_name"])

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

def create_agent(n_actions=None, model=None, ):
    return QAgent(model=model, n_actions=n_actions)

class Experiment(DQN):
    def __init__(self, config, create_env, create_agent):
        super().__init__(config, create_env, create_agent)

    def _create_model(self):
        module=None
        if (not self.config["use_duelling"]):
            module = QMLP(self.obs_dim, self.n_actions, 64)
        else:
            module = DQMLP(self.obs_dim, self.n_actions, 64)

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
