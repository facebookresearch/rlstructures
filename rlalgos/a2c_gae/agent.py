#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
#import rlstructures.logging as logging
from rlstructures import DictTensor
from rlstructures import RL_Agent
import time
from rlstructures.dicttensor import masked_tensor,masked_dicttensor

class RecurrentAgent(RL_Agent):
    def __init__(self,model=None, n_actions=None):
        super().__init__()
        self.model = model
        self.n_actions = n_actions


    def update(self,  state_dict):
        self.model.load_state_dict(state_dict)

    def initial_state(self,agent_info,B):
        return DictTensor({"agent_state":self.model.action_model.initial_state(B),"agent_step":torch.zeros(B).long()})

    def __call__(self, state,observation,agent_info,history=None):
        """
        Executing one step of the agent
        """
        assert not state is None

        initial_state = observation["initial_state"]
        B = observation.n_elems()

        istate=self.initial_state(agent_info,B)
        state=masked_dicttensor(state,istate,initial_state)

        new_z,action_proba = self.model.action_model(state["agent_state"],observation["frame"])

        dist = torch.distributions.Categorical(action_proba)
        action_sampled = dist.sample()

        action_max = action_proba.max(1)[1]
        smask=agent_info["stochastic"].float()
        action=masked_tensor(action_max,action_sampled,agent_info["stochastic"])

        new_state = DictTensor({"agent_state":new_z,"agent_step": state["agent_step"] + 1})
        agent_do = DictTensor(
            {"action": action,"action_probabilities":action_proba}
        )
        return agent_do, new_state

    def call_replay(self,trajectories,t,state):
        """
        Executing one step of the agent
        """
        info=trajectories.info
        trajectories=trajectories.trajectories
        if state is None:
            state=info.truncate_key("agent_state/")

        agent_info=info.truncate_key("agent_info/")
        tslice=trajectories.temporal_index(t)
        observation=tslice.truncate_key("observation/")
        _observation=tslice.truncate_key("_observation/")

        initial_state = observation["initial_state"]
        B = observation.n_elems()
        istate=self.initial_state(agent_info,B).to(initial_state.device)

        state=masked_dicttensor(state,istate,initial_state)
        new_z,action_proba = self.model.action_model(state["agent_state"],observation["frame"])

        diff=(action_proba-tslice.truncate_key("action/")["action_probabilities"]).abs().mean()

        #Check  that there is no computation error: replay is computing the same action probabilities than the batcher agents
        # Note that this problem will happen when using PPO
        # if diff>1e-7:
        #     print("Problem ? ",diff)
        #     #exit()

        critic=self.model.critic_model(state["agent_state"],observation["frame"])
        _critic=self.model.critic_model(new_z,_observation["frame"]).detach()
        new_state = DictTensor({"agent_state":new_z,"agent_step": state["agent_step"] + 1})
        return DictTensor({"critic":critic,"_critic":_critic,"action_probabilities":action_proba}), new_state

class Model(nn.Module):
    def __init__(self,action_model,critic_model):
        super().__init__()
        self.action_model=action_model
        self.critic_model=critic_model

class ActionModel(nn.Module):
    """ The model that computes one score per action
    """
    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.gru = nn.RNNCell(n_hidden,n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, n_actions)
        self.n_hidden=n_hidden
        self.n_actions=n_actions
        self.actf=torch.tanh

    def initial_state(self,B):
        return torch.ones(B,self.n_hidden)

    def forward(self, state,frame):
        frame = self.actf(self.linear(frame))
        z=self.gru(frame,state)
        zz = self.actf(self.linear2(z))
        score_actions=self.linear3(zz)
        return z,torch.softmax(score_actions,dim=-1)

class CriticModel(nn.Module):
    """ The model that computes V(s)
    """
    def __init__(self, n_observations, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear_state = nn.Linear(n_hidden, n_hidden)
        self.linear_z = nn.Linear(n_hidden*2, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)
        self.n_hidden=n_hidden
        self.actf=torch.tanh

    def initial_state(self,B):
        return torch.zeros(B,self.n_hidden)

    def forward(self,state,frame):
        frame =self.actf(self.linear(frame))
        z = self.actf(self.linear_z(torch.cat([frame,state],dim=1)))
        critic = self.linear2(z)
        return critic
