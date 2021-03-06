rlstructures -- mgirating from v0.1 to v0.2
===========================================

Version 0.2 of rlstructures have some critical changes:

From Agent to RL_Agent
----------------------

Policies are now implemented through the RL_Agent class. The two differences are:

* The RL_Agent class has a `initial_state` methods that initialize the state of the agent at reset time (i.e when you call Batcher.reset). It avoids you to handle the state initialization in the `__call__` function.

* The RL_Agent does not return its `old state` anymore, and just provide the `agent_do` and `new_state` as an output

From EpisodeBatcher/Batcher to RL_Batcher
-----------------------------------------

RL_Batcher is the batcher class that works with RL_Agent:

* At construction time:

  * There is no need to specify the `n_slots` arguments anymore

  * One has to provide examples (with n_elems()==1) of `agent_info` and `env_info` that will be sent to the batcher at construction time

  * You can specify the device of the batcher (default is CPU -- see the CPU/GPU tutorial)

* At use time:

  * Only three functions are available: `reset`, `execute` and `get`

* Outputs:

  * The RL_Batcher now outputs a `Trajectories` object composed of `trajectories.info:DictTensor` and `trajectories.trajectories:TemporalDictTensor`

  * `trajectories.info` contains informations that is fixed during the trajectorie: agent_info, env_info and initial agent state

  * `trajectories.trajectories` contains informations generated by the environment (observations), and also actions produced by the Agent

Replay functions
----------------

We now propose a `replay_agent` function that allows to easily repaly an agent over trajectories (e.g for loss computation)
