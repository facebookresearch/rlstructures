#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from rlstructures.env import VecEnv
from rlstructures import DictTensor
import torch

class DeviceEnv:
    def __init__(self,env,from_device,to_device):
        self.env=env
        self.from_device=from_device
        self.to_device=to_device
        self.action_space=self.env.action_space

    def reset(self, env_info=DictTensor({})):
        assert env_info.empty() or env_info.device()==torch.device("cpu"),"env_info must be on CPU"
        o,e=self.env.reset(env_info)
        return o.to(self.to_device),e.to(self.to_device)

    def step(self, policy_output):
        policy_output=policy_output.to(self.from_device)
        (a,b),(c,d)=self.env.step(policy_output)
        return (a.to(self.to_device),b.to(self.to_device)),(c.to(self.to_device),d.to(self.to_device))

    def close(self):
        self.env.close()

    def n_envs(self):
       return self.env.n_envs()
