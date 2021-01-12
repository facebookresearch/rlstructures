#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures import logging
from rlstructures import DictTensor,TemporalDictTensor
import torch

class Agent:
    """
    Describes an agent responsible for producing actions when receiving
    observations and agent states.

    At each time step, and agent receives observations (DictTensor of size B),
    and agent states (DictTensor of size B) that reflect the agent's internal
    state.

    It then returns a triplet:
        agent state when receiving the observation (DictTensor): it is the
            agent state before computing anything. It is mainly used to
            initialize the state of the agent when facing initial states from the environment.
        action (DictTensor): the action + and additional outputs produced by
            the agent
        next agent state (DictTensor): the new state of the agent after all
            the computation. This value will then be provided to the agent at
            the next timestep.
    """
    def __init__(self):
        pass

    def require_history(self):
        """ if True, then the 'history' argument in the __call__ method will contain the set of previous transitions (e.g for transformers based policies)
        """
        return False

    def __call__(self, state:DictTensor, input:DictTensor,user_info:DictTensor,history:TemporalDictTensor = None):
        """ Execute one step of the agent

        :param state: the previous state of the agent, or None if the agent needs to be initialized
        :type state: DictTensor
        :param input: The observation coming from the environment
        :type input: DictTensor
        :param user_info: An additional DictTensor (provided by the user such that the epsilon value in epsilon-greedy policies)
        :type user_info: DictTensor
        :param history: [description], None if require_history()==False or a set of previous transitions (as a TemporalDictTensor) if True
        :type history: TemporalDictTensor, optional
        """
        raise NotImplementedError

    def update(self, info):
        """
        Update the agent. For instance, may update the pytorch model of this agent
        """
        raise NotImplementedError

    def close(self):
        """
        Terminate the agent
        """
        pass
