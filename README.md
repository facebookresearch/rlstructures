# RLStructures

- [RLStructures](#rlstructures)
  - [Overview](#overview)
  - [Learning RLStructures](#learning-rlstructures)
  - [Tutorial](#tutorial)
  - [Algorithms (as examples)](#algorithms-as-examples)
- [USing RLStructures](#using-rlstructures)
- [FAQ](#faq)
  - [License](#license)

## Overview

Classical Deep Learning (DL) algorithms are optimizing a loss function defined over training dataset. It thus usually involve the following training loop:

<img src="images/mlloop.png" width="260">

This is very simple. Moreover, the computation can be easily speed-up by i) using GPUs for the loss and gradient computations, ii) by using multi-processes DataLoaders to deal with large amounts of data. 

*Can we do the same in Reinforcement Learning? This is the objective of RLStructures.*

The main difference between classical DL approaches and Reinforcement Learning ones is 
in the way data are acquired. While in DL batches come from a dataset, in RL, learning data come from the interaction between policies and an environments. Moreover, the nature of the collected information is usually more complex, structured as sequences, and involving multiple variables (e.g observations, actions, internal state of the policies, etc…), particularly when considering complex policies like hierarchical ones, mixtures, etc…. 

RLStructures is a library focused on making the implementation of RL algorithms **as simple as possible**, providing  tools allowing users to write RL learning loops easily. Indeed, RLStructures will take care of simulating the interations between multiple policies and multiple agents at scale and will return a simple data structure on which loss computation is easy. **RLStructures is not a library of RL algorithms** (even if some algorithms are provided to illustrate how it can be used in practical cases.) and can be used in any setting where policies interact with  environments, and when **the user need to do it at scale**, including unsupervised RL, meta-RL, multitask RL, etc…. 

RLStructures is based on three components that have been made as easy as possible to facilitate the use of the library:
* A generic data structure (DictTensor) encoding (batches of) information exchanged between the agent and the environment. In addition, we provide a temporal version of this structure (TemporalDictTensor) encoding (batches of) sequences of DictTensor, the sequences being of various lengths
* An Agent API allowing to implement complex agents (with or without using pytorch  models) with complex outputs and internal states (e.g hierarchical agents, transformers-based agents, …). The API also allows the user to specify which information as to be stored to allow future computations.
* A set of Batchers where a batcher handles the execution of multiple agents over multiple environments and produces as an output a data structure (TemporalDictTensor) that will be easily usable for computation. By using multi-processes batchers, the user will easily scale his/her algorithm to multiple environments, multiple policies on multiple cores. Moreover, since batchers can be executed asynchronuously, the user will be able to colect interactions between agents and environments while executing some other computations in parallel. 

To illustrate these principles, let us illustrate the typical learning loop when using RLStructures:

![ML Learning Loop](images/rlstructloop.png)

Note that multiple GPUs and CPUs can be used (but examples are provided with batchers on CPUs, and learning algorithm on CPU or GPU)

In addition, RLStructures propose:
* A set of classical algorithms (A2C, PPO, DDQN and SAC)
* A simple logger allowing to monitor results on tensorboard, but also to generate CSV files for future analysis


## Learning RLStructures 

Learning RLStructures can be made in a very few hours (from the feedback from multiple users). It involves the following steps (see the rlstructures/tutorial examples)
* Learning about DictTensor and TemporalDictTensor that are the two data structures used everywhere in RLStructures **(15 minutes)** -- see [Data Structures Tutorial](doc/DataStructures.md)

* Learning about mapping a Gym Environment to a RLStructure Environment **(5 minutes)** -- see [Environment](doc/Environments.md)
* Learning about the Agent API allowing one to implement any agent, including recurrent agents **(30 minutes)**.  -- see [Agent Tutorial](doc/Agent.md)
* Learning about creating and using multi-processes batchers (i.e Batcher and EpisodeBatcher) **(15 minutes)**  -- see [Episode Batcher Tutorial](doc/MultiProcessEpisodeBatcher.md) and [Trajectory Batcher Tutorial](doc/MultiProcessTrajectoryBatcher.md)

Now, you can execute a complex policy over an environment in a multi-thread way, and get a simple data structure as an output on which you can compute any complex loss and gradients.

To go deeper in the library, next steps are:
* How to use batchers in non-blocking modes (for instance for computing evaluation metrics while learning using other processes) **(10 mins)**
* How to manage  multiple types of environments and multiple types of agents in a same batcher (e.g goal-oriented policies, multitask environments, managing stochastic and deterministic vesions of a same policy, etc.. ) **(30 mins)**
* How an agent can access its whole history (e.g to implement Transformers-based policies) **(30 mins)**
* Understanding the RLStructures Env API (not needed to develop new algorithms, but may be needed to implement new and fast environments without using openAI gym) **(15 minutes)**

For these additional steps, the best way is to look at the implementation of the examples algorithms.

## Tutorial
In order to learn how to use RLStructures, we provide a simple set of tutorials (see the 'tutorial' directory) to familiarize with the different aspects. 

## Algorithms (as examples)
We provide multiple implementations of RL algorithms as illustrative examples. Note that these algorithms have been benchmarked on simple environments only.
* * A2C for discrete action space (using recurrent or not recurrent architectures) with GAE (i.e including REINFORCE)
* * PPO for discrete action space (using recurrent or not recurrent architectures)
* * Dueling DQN for discrete action space (using not recurrent architectures)
* * SAC for continous action space (using not recurrent architectures)

# USing RLStructures

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{rlstructures,
  author = {Denoyer, Ludovic and Rothermel, Danielle and Martinet, Xavier},
  title = {RLStructures: A library for Reinforcement Learning},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/facebookresearch/rlstructures}},
}
```

* Author: Ludovic Denoyer
* Co-authors: Danielle Rothermel, Xavier Martinet
* Other contributors: many.... 

# FAQ

[Here](https://github.com/facebookresearch/rlstructures/blob/master/FAQ.md)

## License

`rlstructures` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
