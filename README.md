# IMPORTANT NOTE

**October 2021 - SALINA Library**

The principles beyond RLStructures have been extended in the new [SaLinA library](https://github.com/facebookresearch/salina) that can be seen as a simplification step over rlstructures. 


**February 2021 - version 0.2 : Due to recent feedback, the API has been modified improved**

* The new API is not compatible with the old one
* The old API is still working (but printing a deprecated message)
* The v0.1 version of rlstructures is available in the v0.1 github branch
* We encourage users to switch to the v0.2 API which does not need lot of modifications in your current code

Main changes:
* A single Batcher class (instead of two)
* A more clear organization of the information computed by the batcher
* Agents can use seeds for reproducibility issues
* Agents and batchers can work on GPU to speed up the rollouts
* A replay function has been added to allow to replay an Agent over acquired trajectories
  * It greatly facilitate loss functions implementation
* More RL algorithms as examples (including PPO,SAC, REINFORCE, A2C, DQN, DIAYN)
* A growing series of tutorials at [https://ludovic-denoyer.medium.com/](https://ludovic-denoyer.medium.com/)

All these changes are documented in the HTML documentation at [http://facebookresearch.github.io/rlstructures](http://facebookresearch.github.io/rlstructures)


# TL;DR

TL;DR: RLStructures is a lightweight Python library that provides simple APIs as well as data structures that make as few assumptions as possible about the structure of your agent or your task, while allowing for transparently executing multiple policies on multiple environments in parallel (incl. multiple GPUs). It thus facilitates the implementation of RL algorithms while avoiding complex abstractions.

# Why/What?

RL research addresses multiple aspects of RL like hierarchical policies, option-based policies, goal-oriented policies, structured input/output spaces, transformer-based policies, etc. while many available tools are specific to particular settings.

We propose RLStructures as a way to i) simulate multiple policies, multiple models and multiple environments simultaneously at scale ii) define complex loss functions and iii) quickly implement various policy architectures.

The main principle of RLStructures is to allow the user to delegate the sampling of trajectories and episodes to the library so they can spend most of their time on the interesting part of RL research: developing new models and algorithms.

RLStructures is easy to use: it has very few simple interfaces that can be learned in one hour by reading the tutorials. It is not a RL Alorithms catalog, but a library to do RL Research. For illustration purposes it comes with multiple RL algorithms including A2C, PPO, DDQN and SAC.

## Installation

Install from source by running the following inside the repo:
```
pip install .
```

## Learning RLStructures

* [http://facebookresearch.github.io/rlstructures](http://facebookresearch.github.io/rlstructures)

* A must read: tutorials are provided at: [https://ludovic-denoyer.medium.com/](hhttps://ludovic-denoyer.medium.com/)

* The complete documentation is available at [http://facebookresearch.github.io/rlstructures](http://facebookresearch.github.io/rlstructures). The example algorithms are provided in *rlaglos/*.

* A facebook group is also open for discussions : [https://www.facebook.com/groups/834804787067021](https://www.facebook.com/groups/834804787067021)

## Targeted users

RLStructures comes with a set of implemented RL algorithms. But rlstructures does not aim at being a repository of benchmarked RL algorithms (an other RL librairies do that very well). If your objective is to apply state-of-the-art methods on particular environments, then rlstructures is not the best fit. If your objective is to implement new algorithms, then rlstructures is a good fit.

# List of Papers using rlstructures

* [Learning Adaptive Exploration Strategies in Dynamic Environments Through Informed Policy Regularization](https://arxiv.org/abs/2005.02934)
* More to come...


# Citing RLStructures

Please use this bibtex if you want to cite this repository in your publications:

```
    @misc{rlstructures,
        author = {Ludovic Denoyer, Danielle Rothermel and Xavier Martinet},
        title = {{RLStructures - A simple library for RL research}},
        year = {2021},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://gitHub.com/facebookresearch/rlstructures}},
    }

```

## License

`rlstructures` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
