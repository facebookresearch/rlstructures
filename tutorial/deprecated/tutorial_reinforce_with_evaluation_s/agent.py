#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
from rlstructures import DictTensor, masked_tensor, masked_dicttensor
from rlstructures import Agent
import time


class ReinforceAgent(Agent):
    def __init__(self, model=None, n_actions=None):
        super().__init__()
        self.model = model
        self.n_actions = n_actions

    def update(self, state_dict):
        self.model.load_state_dict(state_dict)

    def initial_state(selfl, agent_info, B):
        return DictTensor({})

    def __call__(self, state, observation, agent_info=None, history=None):
        """
        Executing one step of the agent
        """
        # Verify that the batch size is 1
        B = observation.n_elems()
        # We will store the agent step in the trajectories to illustrate how information can be propagated among multiple timesteps
        # We compute one score per possible action
        action_proba = self.model.action_model(observation["frame"])
        baseline = self.model.baseline_model(observation["frame"])
        # We sample an action following the distribution
        dist = torch.distributions.Categorical(action_proba)
        action_sampled = dist.sample()

        # Depending on the agent_info variable that tells us if we are in 'stochastic' or 'deterministic' mode, we keep the sampled action, or compute the action with the max score
        action_max = action_proba.max(1)[1]
        smask = agent_info["stochastic"].float()
        action = masked_tensor(action_max, action_sampled, agent_info["stochastic"])

        new_state = DictTensor({})

        agent_do = DictTensor(
            {
                "action": action,
                "action_probabilities": action_proba,
                "baseline": baseline,
            }
        )

        return agent_do, new_state


class Model(nn.Module):
    def __init__(self, action_model, baseline_model):
        super().__init__()
        self.action_model = action_model
        self.baseline_model = baseline_model


class ActionModel(nn.Module):
    """The model that computes one score per action"""

    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_actions)

    def forward(self, frame):
        z = torch.tanh(self.linear(frame))
        score_actions = self.linear2(z)
        probabilities_actions = torch.softmax(score_actions, dim=-1)
        return probabilities_actions


class BaselineModel(nn.Module):
    """The model that computes V(s)"""

    def __init__(self, n_observations, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)

    def forward(self, frame):
        z = torch.tanh(self.linear(frame))
        critic = self.linear2(z)
        return critic
