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
import numpy as np

class QAgent(Agent):
    """
    Describes a discrete agent based on a model that produces a score for each
    possible action, and an estimation of the value function in the current
    state.
    """

    def __init__(self, model=None,n_actions=None):
        """
        Args:
            model (nn.Module): a module producing a tuple: (actions scores, value)
            n_actions (int): the number of possible actions
        """
        super().__init__()        
        self.model = model
        self.n_actions = n_actions
        
    def update(self,  sd):       
        self.model.load_state_dict(sd)
        
    def __call__(self, state, observation,agent_info=None,history=None):
        """
        Executing one step of the agent
        """
        # Verify that the batch size is 1
            
        initial_state = observation["initial_state"]
        B = observation.n_elems()
        
        if agent_info is None:
            agent_info=DictTensor({"epsilon":torch.zeros(B)})
        
        agent_step = None
        if state is None:
            assert initial_state.all()
            agent_step = torch.zeros(B).long()
        else:
            agent_step = (
                initial_state.float() * torch.zeros(B)
                + (1 - initial_state.float()) * state["agent_step"]
            ).long()

        q = self.model(
            observation["frame"]
        )

        qs,action = q.max(1)
        raction = torch.tensor(np.random.randint(low=0,high=self.n_actions,size=(action.size()[0])))         
        epsilon=agent_info["epsilon"]
        mask=torch.rand(action.size()[0]).lt(epsilon).float()
        action=mask*raction+(1-mask)*action
        action=action.long()


        new_state = DictTensor(
            {"agent_step": agent_step + 1}
        )
       
        agent_do = DictTensor(
            {"action": action, "q": q}
        )

        state = DictTensor({"agent_step": agent_step})

        return state, agent_do, new_state

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
        self.linear_adv= nn.Linear(n_hidden, n_actions)
        self.linear_value = nn.Linear(n_hidden, 1)
        self.n_actions=n_actions

    def forward_common(self, frame):
        z = torch.tanh(self.linear(frame))
        return z

    def forward_value(self,z):
        return self.linear_value(z)

    def forward_advantage(self,z):
        adv=self.linear_adv(z)
        advm=adv.mean(1).unsqueeze(-1).repeat(1,self.n_actions)
        return adv-advm

    def forward(self, state):
        z = self.forward_common(state)
        v = self.forward_value(z)
        adv = self.forward_advantage(z)
        return v+adv


