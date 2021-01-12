#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import rlstructures.logging as logging
from rlstructures import DictTensor
from rlstructures import Agent
import time


class NNAgent(Agent):
    """
    Describes a discrete agent based on a model that produces a score for each
    possible action, and an estimation of the value function in the current
    state.
    """

    def __init__(self,model=None, n_actions=None):
        """
        Args:
            model (nn.Module): a module producing a tuple: (actions scores, value)
            n_actions (int): the number of possible actions
        """
        super().__init__()
        self.model = model
        self.n_actions = n_actions
        self.z_size = self.model.initial_state(1).size()[1]


    def update(self,  state_dict):
        self.model.load_state_dict(state_dict)

    def __call__(self, state, observation,agent_info=None,history=None):
        """
        Executing one step of the agent
        """
        # Verify that the batch size is 1
        initial_state = observation["initial_state"]
        B = observation.n_elems()

        if agent_info is None:
            agent_info=DictTensor({"stochastic":torch.tensor([True]).repeat(B)})

        model_initial_state = self.model.initial_state(B)
        agent_state = None
        agent_step = None
        if state is None:
            assert initial_state.all()
            agent_state = model_initial_state
            agent_step = torch.zeros(B).long()
        else:
            _is = (
                initial_state.float()
                .unsqueeze(-1)
                .repeat(1, model_initial_state.size()[1])
            )
            agent_state = _is * model_initial_state + (1 - _is) * state["agent_state"]
            agent_step = (
                initial_state.float() * torch.zeros(B)
                + (1 - initial_state.float()) * state["agent_step"]
            ).long()

        score_action, value, next_state = self.model(
            agent_state, observation["frame"], observation["last_action"]
        )
        action_proba = torch.softmax(score_action, dim=1)
        dist = torch.distributions.Categorical(action_proba)
        action_sampled = dist.sample()
        action_max = action_proba.max(1)[1]
        smask=agent_info["stochastic"].float()
        action=(action_sampled*smask+(1-smask)*action_max).long()

        new_state = DictTensor(
            {"agent_state": next_state, "agent_step": agent_step + 1}
        )

        agent_do = DictTensor(
            {"action": action, "action_probabilities": action_proba}
        )

        state = DictTensor({"agent_state": agent_state, "agent_step": agent_step})
        return state, agent_do, new_state

class MLPAgentModel(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_actions)
        self.linear_v = nn.Linear(n_hidden, 1)

    def initial_state(self, batch_size):
        d = next(self.parameters()).device
        return torch.zeros(batch_size, 1).to(d)

    def forward(self, state, frame, last_action):
        z = torch.tanh(self.linear(frame))
        score_actions = self.linear2(z)
        value = self.linear_v(z)
        return score_actions, value, self.initial_state(frame.size()[0]) + 1.0


class GRUAgentModel(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_hidden, n_actions)
        self.linear_obs = nn.Linear(n_observations, n_hidden)
        self.linear_action = nn.Linear(n_actions, n_hidden)
        self.gru = nn.GRUCell(2 * n_hidden, n_hidden)
        self.linear_v = nn.Linear(n_hidden, 1)
        self.n_hidden = n_hidden
        self.n_actions = n_actions

    def initial_state(self, batch_size):
        d = next(self.parameters()).device
        return torch.zeros(batch_size, self.n_hidden).to(d)

    def forward(self, state, frame, last_action):
        frame = torch.relu(self.linear_obs(frame))
        B = frame.size()[0]
        oh = torch.zeros(B, self.n_actions).to(frame.device)
        oh[torch.arange(B).to(frame.device), last_action] = 1.0
        oh = self.linear_action(oh)

        z = self.gru(torch.cat([frame, oh], dim=1), state)
        score_actions = self.linear(z)
        value = self.linear_v(z)
        return score_actions, value, z
