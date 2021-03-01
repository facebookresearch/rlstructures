#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn

# import rlstructures.logging as logging
from rlstructures import DictTensor
from rlstructures import RL_Agent
import time
import numpy as np


class QAgent(RL_Agent):
    def __init__(self, model=None, n_actions=None):
        super().__init__()
        self.model = model
        self.n_actions = n_actions

    def update(self, sd):
        self.model.load_state_dict(sd)

    def initial_state(self, agent_info, B):
        return DictTensor({})

    def __call__(self, state, observation, agent_info=None, history=None):
        B = observation.n_elems()

        agent_step = None
        q = self.model(observation["frame"])
        qs, action = q.max(1)
        raction = torch.tensor(
            np.random.randint(low=0, high=self.n_actions, size=(action.size()[0]))
        )
        epsilon = agent_info["epsilon"]
        r = torch.rand(action.size()[0])
        mask = r.lt(epsilon).float()
        action = mask * raction + (1 - mask) * action
        action = action.long()

        agent_do = DictTensor({"action": action, "q": q})
        return agent_do, DictTensor({})


class DQMLP(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear_adv = nn.Linear(n_hidden, n_actions)
        self.linear_value = nn.Linear(n_hidden, 1)
        self.n_actions = n_actions

    def forward_common(self, frame):
        z = torch.tanh(self.linear(frame))
        return z

    def forward_value(self, z):
        return self.linear_value(z)

    def forward_advantage(self, z):
        adv = self.linear_adv(z)
        advm = adv.mean(1).unsqueeze(-1).repeat(1, self.n_actions)
        return adv - advm

    def forward(self, state):
        z = self.forward_common(state)
        v = self.forward_value(z)
        adv = self.forward_advantage(z)
        return v + adv
