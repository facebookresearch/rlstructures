# (January, 2021)

* Initial Release (v0.1)

# (February 2021)

* [REFACTOR] Release of the v0.2 version of rlstructures
* [DOCUMENTATION] Release of tutorials: introduction, principles, reinforce, diayn, a2c and GPU/CPU: https://ludovic-denoyer.medium.com/

# (March 2021)

* [FEATURES] agent_state/... and _agent/state/... are not available through trajectories.trajectories after a batcher.get
* [FIX] correct rlstructures version in setup.py
* [FIX] correct PPO implementation (not correctly updated from v0.1 -. v0.2)
* [FIX] correct assert in DictTensor.set to allow to use the function when the DictTensor is empty
* [FEATURES] add DictTensor.unset
