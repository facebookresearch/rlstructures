Batchers
========

Batchers are objects allowing the execution of multiple policies over multiple environments, using multiple processes. We provide two batchers:
1) `EpisodeBatcher` to acquire complete episodes (until a `done` is reached)
2) `Batcher` to acquire the `n` next timesteps over multiple environments

Examples about how to use batchers are given in the other sections.

**Important**: All the information (e.g `DictTensor`) produced by both the `Agent` and the `Env` will be available to the user in the returned `TemporalDictTensor`

Parallelization Schema
----------------------

The generic parallelization schema is illustrated in the following picture.

.. image:: https://raw.githubusercontent.com/facebookresearch/rlstructures/main/docs/images/batchers.jpg?token=ABNXVXPVRMSMY5XGYBMOUILAA725Q
  :width: 1024
  :alt: Parallelization Schema

* One batcher creates multiple processes

* Each process contains a copy of the `Agent` and a copy of the `rlstructures.VecEnv` (the copy are made through `create_agent` and `create_env` functions that are arguments at the batcher creation)

  * In our case, each Agent as its own copy of the pytorch model (note that it can be a shared_memory model to avoid to use extra memory)

* Each `VecEnv` corresponds to multiple simple environments

* At `execute`, the processes start to acquire information by simulating the interaction betweeh the agent and the environments

* At `get`, the information collected by the processes are merged to a `TemporalDictTensor` that is the output of `get`

* The call of `Batcher.update` will call `Agent.update` in all the processes to typically update the model of each Agent
