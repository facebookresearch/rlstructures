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
from rlstructures.dicttensor import masked_tensor, masked_dicttensor


class AtariAgent(RL_Agent):
    def __init__(self, model=None, n_actions=None):
        super().__init__()
        self.model = model
        self.n_actions = n_actions

    def update(self, state_dict):
        self.model.load_state_dict(state_dict)

    def initial_state(self, agent_info, B):
        return DictTensor({"agent_step": torch.zeros(B).long()})

    def __call__(self, state, observation, agent_info, history=None):
        """
        Executing one step of the agent
        """
        assert not state is None
        initial_state = observation["initial_state"]
        B = observation.n_elems()

        istate = self.initial_state(agent_info, B)
        state = masked_dicttensor(state, istate, initial_state)

        action_proba = self.model.action_model(observation["frame"])
        dist = torch.distributions.Categorical(action_proba)
        action_sampled = dist.sample()

        action_max = action_proba.max(1)[1]
        smask = agent_info["stochastic"].float()
        action = masked_tensor(action_max, action_sampled, agent_info["stochastic"])

        new_state = DictTensor({"agent_step": state["agent_step"] + 1})
        agent_do = DictTensor({"action": action, "action_probabilities": action_proba})
        return agent_do, new_state

    def call_replay(self, trajectories, t, state):
        """
        Executing one step of the agent
        """
        info = trajectories.info
        trajectories = trajectories.trajectories
        if state is None:
            state = info.truncate_key("agent_state/")

        agent_info = info.truncate_key("agent_info/")
        tslice = trajectories.temporal_index(t)
        observation = tslice.truncate_key("observation/")
        _observation = tslice.truncate_key("_observation/")

        initial_state = observation["initial_state"]
        B = observation.n_elems()

        istate = self.initial_state(agent_info, B).to(initial_state.device)
        state = masked_dicttensor(state, istate, initial_state)
        action_proba = self.model.action_model(observation["frame"])

        diff = (
            (action_proba - tslice.truncate_key("action/")["action_probabilities"])
            .abs()
            .mean()
        )
        # Check  that there is no computation error: replay is computing the same action probabilities than the batcher agents
        if diff > 1e-5:
            print(diff)
            print("Problem ?")
            exit()

        critic = self.model.critic_model(observation["frame"])
        _critic = self.model.critic_model(observation["frame"]).detach()
        new_state = DictTensor({"agent_step": state["agent_step"] + 1})
        return (
            DictTensor(
                {
                    "critic": critic,
                    "_critic": _critic,
                    "action_probabilities": action_proba,
                }
            ),
            new_state,
        )


class Model(nn.Module):
    def __init__(self, action_model, critic_model):
        super().__init__()
        self.action_model = action_model
        self.critic_model = critic_model


class ActionModel(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super().__init__()

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
        return torch.softmax(x, dim=-1)

    def features_size(self):
        a = self.features(torch.zeros(1, *self.inut_shape[1:]))
        a = a.view(1, -1).size(1)
        return a


class CriticModel(nn.Module):
    """The model that computes V(s)"""

    def __init__(self, inputs_shape):
        super().__init__()

        self.inut_shape = inputs_shape

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512), nn.ReLU(), nn.Linear(512, 1)
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
