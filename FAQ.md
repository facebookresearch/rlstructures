Overview
========

Classical Deep Learning (DL) algorithms are optimizing a loss function defined over training dataset. The implementation of such a training loop is quite simple.
Moreover, the computation can be easily speed-up by i) using GPUs for the loss and gradient computations, ii) by using multi-processes DataLoaders to deal with large amounts of data.

*Can we do the same in Reinforcement Learning? This is the objective of RLStructures.*

The main difference between classical DL approaches and Reinforcement Learning ones is
in the way data are acquired. While in DL batches come from a dataset, in RL, learning data come from the interaction between policies and an environments. Moreover, the nature of the collected information is usually more complex, structured as sequences, and involving multiple variables (e.g observations, actions, internal state of the policies, etc…), particularly when considering complex policies like hierarchical ones, mixtures, etc….

RLStructures is a library focused on making the implementation of RL algorithms **as simple as possible**, providing  tools allowing users to write RL learning loops easily. Indeed, RLStructures will take care of simulating the interations between multiple policies and multiple agents at scale and will return a simple data structure on which loss computation is easy. **RLStructures is not a library of RL algorithms** (even if some algorithms are provided to illustrate how it can be used in practical cases.) and can be used in any setting where policies interact with  environments, and when **the user need to do it at scale**, including unsupervised RL, meta-RL, multitask RL, etc….

RLStructures is based on three components that have been made as easy as possible to facilitate the use of the library:

* A generic data structure (DictTensor) encoding (batches of) information exchanged between the agent and the environment. In addition, we provide a temporal version of this structure (TemporalDictTensor) encoding (batches of) sequences of DictTensor, the sequences being of various lengths
* An Agent API allowing to implement complex agents (with or without using pytorch  models) with complex outputs and internal states (e.g hierarchical agents, transformers-based agents, …). The API also allows the user to specify which information as to be stored to allow future computations.
* A set of Batchers where a batcher handles the execution of multiple agents over multiple environments and produces as an output a data structure (TemporalDictTensor) that will be easily usable for computation. By using multi-processes batchers, the user will easily scale his/her algorithm to multiple environments, multiple policies on multiple cores. Moreover, since batchers can be executed asynchronuously, the user will be able to colect interactions between agents and environments while executing some other computations in parallel.

Note that multiple GPUs and CPUs can be used (but examples are provided with batchers on CPUs, and learning algorithm on CPU or GPU)

Moreover, RLStructures proposes:

* A set of classical algorithms (A2C, PPO, DDQN and SAC)
* A simple logger allowing to monitor results on tensorboard, but also to generate CSV files for future analysis
