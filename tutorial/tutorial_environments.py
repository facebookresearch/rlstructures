#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
from gym.utils import seeding
from gym.spaces import Discrete

# Defining a custom environment.
# 1. It is a gym.Env environment
# 2. (but observation_space can be empty)
# 3. The observation can be a list/np.array or a dictionnary of list/np.array
# 4. The reset function may receive a env_info arguments as a dictionnary of list/np.array
class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(2)

    def seed(self, seed=None):
        print("Seed = %d" % seed)
        self.np_random, seed = seeding.np_random(seed)

    def reset(self, env_info={}):
        self.x = self.np_random.rand() * 2.0 - 1.0
        self.identifier = self.np_random.rand()
        return {"x": self.x, "identifier": self.identifier}

    def step(self, action):
        if action == 0:
            self.x -= 0.3
        else:
            self.x += 0.3

        return (
            {"x": self.x, "identifier": self.identifier},
            self.x,
            self.x < -1 or self.x > 1,
            {},
        )


# Now, one can use a wrapper to transform this gym.Env to a rlstructures.VecEnv
# 1. A VecEnv corresponds to env.n_envs() environnements that are running simultaneously
# 2. VecEnv receives a dictTensor d as action where d.n_elems()<=env.n_envs()
# 3. VecEnv returns a DictTensor obs as an observation. This observation contains multiple "field".
# e.g obs["reward"] is the reward signal, obs["initial_state"] tells if it is the first state of a new episode, ...
# Actually, since N <= env.n_envs() environments are still running at timestep t, the VecEnv.reset()  and VecEnv.step(...) methods also returns the list of envs that are still running
#
# Example: (obs,who_was_running),(obs2,who_is_still_running) = env.step(action)
# * obs is the observation (at t) coming from the environments that were running at t-1
# * who_was_running is the list of environnments still running at time t-1. Note that who_was_running.size()[0]=obs.n_elems()
# * obs2 is the observation (at t) from the environments that are still running at time t (i.e obs2 is a subset of obs)
# * who_is_still_running is the list of environments running at time t

from rlstructures.env_wrappers import GymEnv
from rlstructures import DictTensor
import torch

envs = [MyEnv() for k in range(4)]
env = GymEnv(envs, seed=80)

# Each instance of the gym.Env will be initialized with seed+i such that the multiple instances will have different seeds

# Interaction with the environment is easy, but made by using DictTensor

obs, who_is_still_running = env.reset()
print(obs)
n_running = who_is_still_running.size()[0]
while n_running > 0:  # While some envs are still running
    action = DictTensor({"action": torch.tensor([0]).repeat(n_running)})
    (obs, who_was_running), (obs2, who_is_still_running) = env.step(action)
    n_running = who_is_still_running.size()[0]
    print(obs2)

# Note that gym wrappers work with continuous and discrete action spaces, but may not with environments where the action space is more complicated.
# If you are facing gym envs with a complex action space, you may develop your own wrapper
# A good starting point is the rlstructures.GymEnv code which is very simple can be used to define a new wrapper
# All the other rlstuctures components will work with complex action spaces without modifications
