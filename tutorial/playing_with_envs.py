#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from rlstructures.env_wrappers import GymEnv
from rlstructures import DictTensor
import torch
import gym

envs = [gym.make("CartPole-v0") for k in range(4)]
env = GymEnv(envs, seed=80)

obs, who_is_still_running = env.reset()
print(obs)
n_running = who_is_still_running.size()[0]
while n_running > 0:  # While some envs are still running
    action = DictTensor({"action": torch.tensor([0]).repeat(n_running)})
    (obs, who_was_running), (obs2, who_is_still_running) = env.step(action)
    n_running = who_is_still_running.size()[0]
    print(obs)
