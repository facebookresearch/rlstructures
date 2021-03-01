#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures.env_wrappers import GymEnv,DeviceEnv
from rlalgos.tools import weight_init
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from rlalgos.reinforce.agent import ReinforceAgent, ActionModel, BaselineModel, Model
from rlalgos.reinforce_device.reinforce import Reinforce
import gym
from gym.wrappers import TimeLimit
from rlstructures import RL_Agent_CheckDevice
# We write the 'create_env' and 'create_agent' function in the main file to allow these functions to be used with pickle when creating the batcher processes

def create_gym_env(env_name):
    return gym.make(env_name)

# Create a rlstructures.VecEnv from multiple gym.Env, limiting the number of steps
def create_env(n_envs, env_name=None, max_episode_steps=None, device=None,seed=None):
    envs = []
    for k in range(n_envs):
        e = create_gym_env(env_name)
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return DeviceEnv(GymEnv(envs, seed),from_device=torch.device("cpu"),to_device=device)

# Create a rlstructures.Agent
def create_agent(model, n_actions=1,device=None,copy_model=True):
    print("create agent on ",device," with model copy==",copy_model)
    if copy_model:
        model=copy.deepcopy(model)

    agent=ReinforceAgent(model=model.to(device), n_actions=n_actions)
    return RL_Agent_CheckDevice(agent,device)

class Experiment(Reinforce):
    def __init__(self, config, create_env, create_agent):
        super().__init__(config, create_env, create_agent)

    def _create_model(self):
        action_model = ActionModel(self.obs_dim, self.n_actions, 16)
        baseline_model = BaselineModel(self.obs_dim, 16)
        return Model(action_model, baseline_model)


if __name__ == "__main__":
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
        "time_limit": 20,
        "lr": 0.01,
        "discount_factor": 0.9,
        "baseline_coef": 0.1,
        "entropy_coef": 0.01,
        "reinforce_coef": 1.0,
        "evaluation_mode": "stochastic",
        "logdir": "./results",
        "learner_device":torch.device("cuda:0"),
        "batcher_device":torch.device("cuda:1"),
        "evaluation_device":torch.device("cpu")
    }
    exp = Experiment(config, create_env, create_agent)
    exp.run()
