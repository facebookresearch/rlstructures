Provided Algorithms
===================

We provide multiple RL algorithms as examples.

1) A2C with General Advantage Estimator
2) PPO with discrete actions
3) Double Duelling Q-Learning + Prioritized Experience Replay
3bis) A simpler DQN implementation (as an example)
4) SAC for continuous actions
5) REINFORCE
6) REINFORCE DIAYN (see https://arxiv.org/abs/1802.06070)

The algorithms can be used as examples to implement your own algorithms.

Typical execution is `OMP_NUM_THREADS=1 PYTHONPATH=rlstructures python rlstructures/rlalgos/reinforce/main_reinforce.py`

Note that all algorithms produced a tensorboard and a CSV output (see `config["logdir"]` in the main file)
