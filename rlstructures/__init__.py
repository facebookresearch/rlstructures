__version__ = "0.2"
__deprecated_message__ = False
import sys
from rlstructures.core import (
    masked_tensor,
    masked_dicttensor,
    DictTensor,
    TemporalDictTensor,
    Trajectories,
)
from rlstructures.rl_batchers.agent import (
    RL_Agent,
    RL_Agent_CheckDevice,
    replay_agent_stateless,
    replay_agent,
)
from rlstructures.env import VecEnv
from rlstructures.rl_batchers import RL_Batcher


# Deprecated import == Old version of rlstructures
from rlstructures.deprecated.agent import Agent

import rlalgos.logger

sys.modules["rlstructures.logger"] = rlalgos.logger

import rlalgos.tools

sys.modules["rlstructures.tools"] = rlalgos.tools

import rlstructures.core

sys.modules["rlstructures.dicttensor"] = rlstructures.core

import rlstructures.deprecated.logging

sys.modules["rlstructures.logging"] = rlstructures.deprecated.logging

import rlstructures.deprecated.batchers

sys.modules["rlstructures.batchers"] = rlstructures.deprecated.batchers

import rlalgos.deprecated.template_exp

sys.modules["rlalgos.template_exp"] = rlalgos.deprecated.template_exp
