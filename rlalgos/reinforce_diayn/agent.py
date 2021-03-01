#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
from rlstructures import DictTensor, masked_tensor, masked_dicttensor
from rlstructures import RL_Agent
import time


class DIAYNAgent(RL_Agent):
    def __init__(self, model=None, n_actions=None):
        super().__init__()
        self.model = model
        self.n_actions = n_actions

    def update(self, state_dict):
        self.model.load_state_dict(state_dict)

    def require_history(self):
        return False

    def initial_state(self, agent_info, B):
        return DictTensor({})

    def __call__(self, state, observation, agent_info=None, history=None):
        """
        Executing one step of the agent
        """
        assert state.empty()

        B = observation.n_elems()

        idx_policy = agent_info["idx_policy"]
        action_proba = self.model.action_model(observation["frame"], idx_policy)
        baseline = self.model.baseline_model(observation["frame"], idx_policy)

        dist = torch.distributions.Categorical(action_proba)
        action_sampled = dist.sample()

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


class DIAYNModel(nn.Module):
    def __init__(self, action_model, baseline_model):
        super().__init__()
        self.action_model = action_model
        self.baseline_model = baseline_model


class DIAYNActionModel(nn.Module):
    """The model that computes one score per action"""

    def __init__(self, n_observations, n_actions, n_hidden, n_policies):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_actions * n_policies)
        self.n_policies = n_policies
        self.n_actions = n_actions

    def forward(self, frame, idx_policy):
        z = torch.tanh(self.linear(frame))
        score_actions = self.linear2(z)
        s = score_actions.size()
        score_actions = score_actions.reshape(s[0], self.n_policies, self.n_actions)
        score_actions = score_actions[torch.arange(s[0]), idx_policy]
        probabilities_actions = torch.softmax(score_actions, dim=-1)
        return probabilities_actions


class DIAYNBaselineModel(nn.Module):
    """The model that computes V(s)"""

    def __init__(self, n_observations, n_hidden, n_policies):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_policies)
        self.n_policies = n_policies

    def forward(self, frame, idx_policy):
        z = torch.tanh(self.linear(frame))
        critic = self.linear2(z)
        critic = critic.reshape(critic.size()[0], self.n_policies, 1)
        critic = critic[torch.arange(critic.size()[0]), idx_policy]
        return critic
