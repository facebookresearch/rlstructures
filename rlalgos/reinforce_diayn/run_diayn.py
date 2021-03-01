#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures.env_wrappers import GymEnv
from rlalgos.tools import weight_init
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from rlalgos.reinforce_diayn.agent import (
    DIAYNAgent,
    DIAYNActionModel,
    DIAYNBaselineModel,
    DIAYNModel,
)
from rlalgos.reinforce_diayn.reinforce_diayn import Reinforce
import gym
from gym.wrappers import TimeLimit

# We write the 'create_env' and 'create_agent' function in the main file to allow these functions to be used with pickle when creating the batcher processes
def create_gym_env(env_name):
    return gym.make(env_name)


# Create a rlstructures.VecEnv from multiple gym.Env, limiting the number of steps
def create_env(n_envs, env_name=None, max_episode_steps=None, seed=None):
    envs = []
    for k in range(n_envs):
        e = create_gym_env(env_name)
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnv(envs, seed)


# Create a rlstructures.Agent
def create_agent(model, n_actions=1):
    return DIAYNAgent(model=model, n_actions=n_actions)


class Experiment(Reinforce):
    def __init__(self, config, create_env, create_agent):
        super().__init__(config, create_env, create_agent)

    def _create_model(self):
        action_model = DIAYNActionModel(
            self.obs_dim, self.n_actions, 16, self.config["n_policies"]
        )
        baseline_model = DIAYNBaselineModel(self.obs_dim, 16, self.config["n_policies"])
        return DIAYNModel(action_model, baseline_model)

    def _create_discriminator(self):
        classifier = nn.Linear(self.obs_dim, self.config["n_policies"])
        classifier.apply(weight_init)
        return classifier


if __name__ == "__main__":
    print(
        "DISCLAIMER: DIAYN is just provided as an example. It has not been tested deeply !!"
    )
    # We use spawn mode such that most of the environment will run in multiple processes
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    config = {
        "env_name": "CartPole-v0",
        "n_envs": 4,
        "max_episode_steps": 100,
        "env_seed": 42,
        "n_processes": 4,
        "n_evaluation_processes": 2,
        "n_evaluation_envs": 128,
        "time_limit": 3600,
        "lr": 0.01,
        "lr_discriminator": 0.01,
        "discount_factor": 0.9,
        "baseline_coef": 0.1,
        "discriminator_coef": 1.0,
        "entropy_coef": 0.01,
        "reinforce_coef": 1.0,
        "evaluation_mode": "stochastic",
        "logdir": "./results",
        "n_policies": 5,
    }
    exp = Experiment(config, create_env, create_agent)
    exp.run()
