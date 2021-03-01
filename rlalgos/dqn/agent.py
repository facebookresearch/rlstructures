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


class QMLP(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_actions)

    def forward(self, frame):
        z = torch.tanh(self.linear(frame))
        score_actions = self.linear2(z)
        return score_actions


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


class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        a = self.features(torch.zeros(1, *self.inut_shape[1:]))
        a = a.view(1, -1).size(1)
        return a


class DuelingCnnDQN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(DuelingCnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512), nn.ReLU(), nn.Linear(512, num_outputs)
        )

        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape[1:])).view(1, -1).size(1)
