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
from rlstructures.dicttensor import masked_tensor,masked_dicttensor

class RecurrentAgent(Agent):
    def __init__(self,model=None, n_actions=None):
        super().__init__()
        self.model = model
        self.n_actions = n_actions


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

        # Create the initial state of the recurrent policy
        agent_initial=self.model.initial_state(B)
        if (state is None): # If the batcher is starting
            state=DictTensor({"agent_state":agent_initial,"agent_step":torch.zeros(B).long()})
        else:
            #Maybe some observations are initial states of new episodes. For these state, we must initialize the internal state of the policy
            istate=DictTensor({"agent_state":agent_initial,"agent_step":torch.zeros(B).long()})
            state=masked_dicttensor(istate,state,initial_state)


        new_z,action_proba = self.model(state["agent_state"],observation["frame"])

        #We sample an action following the distribution
        dist = torch.distributions.Categorical(action_proba)
        action_sampled = dist.sample()

        #Depending on the agent_info variable that tells us if we are in 'stochastic' or 'deterministic' mode, we keep the sampled action, or compute the action with the max score
        action_max = action_proba.max(1)[1]
        smask=agent_info["stochastic"].float()
        action=masked_tensor(action_max,action_sampled,agent_info["stochastic"])


        new_state = DictTensor({"agent_state":new_z,"agent_step": state["agent_step"] + 1})

        agent_do = DictTensor(
            {"action": action, "action_probabilities": action_proba}
        )

        return state, agent_do, new_state

class AgentModel(nn.Module):
    """ The model that computes one score per action
    """
    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear_state = nn.Linear(n_hidden, n_hidden)
        self.linear_z = nn.Linear(n_hidden*2, n_hidden)

        self.linear2 = nn.Linear(n_hidden, n_actions)
        self.n_hidden=n_hidden

    def initial_state(self,B):
        return torch.zeros(B,self.n_hidden)

    def forward(self, state,frame):
        frame = torch.tanh(self.linear(frame))
        state=torch.tanh(self.linear_state(state))
        z=torch.tanh(self.linear_z(torch.cat([frame,state],dim=1)))
        score_actions = self.linear2(z)
        probabilities_actions = torch.softmax(score_actions,dim=-1)
        return z,probabilities_actions

class BaselineModel(nn.Module):
    """ The model that computes V(s)
    """
    def __init__(self, n_observations, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear_state = nn.Linear(n_hidden, n_hidden)
        self.linear_z = nn.Linear(n_hidden*2, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)


    def forward(self,state, frame):
        frame = torch.tanh(self.linear(frame))
        state=torch.tanh(self.linear_state(state))
        z=torch.tanh(self.linear_z(torch.cat([frame,state],dim=1)))
        critic = self.linear2(z)
        return z,critic
