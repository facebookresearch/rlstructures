#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import time
import torch
import rlstructures.logging as logging
from rlstructures import DictTensor

class VecEnv:
    """
    An VecEnvironment corresponds to multiple 'gym' environments (i.e a batch)
    that are running simultaneously.

    At each timestep, upon the B environments, a subset B' of envs are running
    (since some envs may have stopped).

    So each observation returned by the VecEnv is a DictTensor of size B'. To
    mark which environments that are still running, the observation is returned
    with a mapping vector of size B'. e.g [0,2,5] means that the observation 0
    corresponds to the env 0, the observation 1 corresponds to env 2, and the
    observation 3 corresponds to env 5.

    Finally, when running a step (at time t) method (over B' running envs), the
    agent has to provide an action (DictTensor) of size B'. The VecEnv will return
    the next observation (time t+1) (size B'). But some of the B' envs may have
    stopped at t+1, such that actually only B'' envs are still running. The
    step method will thus also return a B'' observation (and corresponding
    mapping).

    The return of the step function is thus:
        ((DictTensor of size B', tensor of size B'),
        (Dicttensor of size B'', mapping vector if size B''))
    """

    def __init__(self):
        pass

    def reset(self,env_info:DictTensor=DictTensor({})):
        """ reset the environments instances

        :param env_info: a DictTensor of size n_envs, such that each value will be transmitted to each environment instance
        :type env_info: DictTensor, optional
        """        
        pass

    def step(self, policy_output:DictTensor)-> [[DictTensor,torch.Tensor],[DictTensor,torch.Tensor]]:
        """ Execute one step over alll the running environment instances

        :param policy_output: the output given by the policy 
        :type policy_output: DictTensor
        :return: see general description
        :rtype: [[DictTensor,torch.Tensor],[DictTensor,torch.Tensor]]
        """                
        raise NotImplementedError

    def close(self):
        """Terminate the environment
        """
        raise NotImplementedError

    def n_envs(self)->int:
        """ Returns the number of environment instances contained in this env        
        :rtype: int
        """        
        return self.reset()[0].n_elems()

