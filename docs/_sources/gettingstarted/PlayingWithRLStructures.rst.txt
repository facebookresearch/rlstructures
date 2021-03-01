Playing with rlstructures
=========================

We propose some examples of Batcher uses to better understand how it works. The python file is https://github.com/facebookresearch/rlstructures/blob/main/tutorial/playing_with_rlstructures.py


Blocking / non-Blocking batcher execution
-----------------------------------------

The `batcher.get` function can be executed in `batcher.get(blocking=True)` or `batcher.get(blocking=False)` modes.

* In the first mode `blocking=True`, the progam will wait the batcher to end its acquisition and will return trajectories

* In the second mode `blocking=False`,the batcher will return `None,None` is the acquisition is not finished. It thus allows to perform other computation without waiting the batcher to finished

Replaying an agent over an acquired trajectory
----------------------------------------------

When trajectories have been acquired, then the autograd graph is not avaialbe (i.e batcher are launched in `require_grad=False` mode).
It is important to be able to recompute the agent steps on these trajectories.

We provide the `replay_agent` function to facilitate this `replay`. An example is given in https://github.com/facebookresearch/rlstructures/blob/main/rlalgos/reinforce

Some other examples of use are given in the A2C and DQN implementations.
