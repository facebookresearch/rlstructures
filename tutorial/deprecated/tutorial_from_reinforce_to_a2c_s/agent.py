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
from rlstructures import S_Agent
import time
from rlstructures.dicttensor import masked_tensor, masked_dicttensor


class ReinforceAgent(S_Agent):
    def __init__(self, model=None, n_actions=None):
        super().__init__()
        self.model = model
        self.n_actions = n_actions

    def update(self, state_dict):
        self.model.load_state_dict(state_dict)

    def initial_state(self, agent_info, B):
        return DictTensor({})

    def __call__(self, state, observation, agent_info=None, history=None):
        """
        Executing one step of the agent
        """
        # Verify that the batch size is 1
        B = observation.n_elems()

        # We compute one score per possible action
        action_proba = self.model.action_model(observation["frame"])
        # We sample an action following the distribution
        dist = torch.distributions.Categorical(action_proba)
        action_sampled = dist.sample()

        # Depending on the agent_info variable that tells us if we are in 'stochastic' or 'deterministic' mode, we keep the sampled action, or compute the action with the max score
        action_max = action_proba.max(1)[1]
        smask = agent_info["stochastic"].float()
        action = masked_tensor(action_max, action_sampled, agent_info["stochastic"])

        new_state = DictTensor({})

        agent_do = DictTensor({"action": action})

        return agent_do, new_state

    def call_replay(self, trajectories, info, t, last_call_state):
        """
        Executing one step of the agent
        """
        tslice = trajectories.temporal_index(t)
        observation = tslice.truncate_key("observation/")
        _observation = tslice.truncate_key("_observation/")
        critic = self.model.critic_model(observation["frame"])
        _critic = self.model.critic_model(_observation["frame"])
        action_probabilities = self.model.action_model(observation["frame"])
        return (
            DictTensor(
                {
                    "critic": critic,
                    "_critic": _critic,
                    "action_probabilities": action_probabilities,
                }
            ),
            None,
        )


class Model(nn.Module):
    def __init__(self, action_model, critic_model):
        super().__init__()
        self.action_model = action_model
        self.critic_model = critic_model


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


class CriticModel(nn.Module):
    """The model that computes V(s)"""

    def __init__(self, n_observations, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)

    def forward(self, frame):
        z = torch.tanh(self.linear(frame))
        critic = self.linear2(z)
        return critic
