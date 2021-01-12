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
import math
import torch.nn.functional as F
from torch import distributions as pyd

class SACAgent(Agent):
    """
    Describes a discrete agent based on a model that produces a score for each
    possible action, and an estimation of the value function in the current
    state.
    """

    def __init__(self,policy=None, action_dim=None):
        """
        Args:
            model (nn.Module): a module producing a tuple: (actions scores, value)
            n_actions (int): the number of possible actions
        """
        super().__init__()       
        self.model = policy
        self.action_dim = action_dim

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


        agent_step = None
        if state is None:
            assert initial_state.all()
            agent_step = torch.zeros(B).long()
        else:
            agent_step = (
                initial_state.float() * torch.zeros(B)
                + (1 - initial_state.float()) * state["agent_step"]
            ).long()

        _mean,_var = self.model(observation["frame"])
        _id = torch.eye(self.action_dim).unsqueeze(0).repeat(B, 1, 1)
        # _nvar = _var.unsqueeze(-1).repeat(1, 1, self.action_dim)
        # _nvar = _nvar * _id
        distribution=torch.distributions.Normal(_mean, _var)
        action_sampled=distribution.sample()
        action_max = _mean
        smask=agent_info["stochastic"].float().unsqueeze(-1).repeat(1,self.action_dim)
        action=(action_sampled*smask+(1.0-smask)*action_max)
        new_state = DictTensor(
            {"agent_step": agent_step + 1}
        )

        agent_do = DictTensor(
            {"action": action, "mean":_mean,"std":_var}
        )
        state = DictTensor({"agent_step": agent_step})
        return state, agent_do, new_state

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, log_std_bounds=[-5, 2]):
        super().__init__()
        self.log_std_bounds = log_std_bounds        

    def forward(self, mu,log_std):
        
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)
        std = log_std.exp()
        mu = torch.tanh(mu)
        return mu,std



class SACPolicy(nn.Module):
    def __init__(self, n_observations, action_dim, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear_mean = nn.Linear(n_hidden, action_dim)
        self.linear_std = nn.Linear(n_hidden, action_dim)
        self.dg=DiagGaussianActor()
        
    def forward(self, frame):
        z = torch.relu(self.linear(frame))
        mean = self.linear_mean(z)
        std = self.linear_std(z)
        return self.dg(mean,std)
        
class SACQ(nn.Module):
    def __init__(self, n_observations, action_dim, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear_2 = nn.Linear(action_dim, n_hidden)
        self.linear_q = nn.Linear(n_hidden*2, n_hidden)
        self.linear_qq = nn.Linear(n_hidden, 1)
        
    def forward(self, frame, action):
        zf = torch.relu(self.linear(frame))
        za = torch.relu(self.linear_2(action))
        q = torch.relu(self.linear_q(torch.cat([zf,za],dim=1)))
        q = self.linear_qq(q)
        return q

