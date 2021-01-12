# TL;DR

TL;DR: RLStructures is a lightweight Python library that provides simple APIs as well as data structures that make as little assumptions as possible on the structure of your agent or your task while allowing for transparently executing multiple policies on multiple environments in parallel (incl. multiple GPUs). It thus facilitates the implementation of RL algorithms while avoiding complex abstractions.

# Why/What?

RL research addresses multiple aspects of RL like hierarchical policies, option-based policies, goal-oriented policies, structured input/output spaces, transformers-based policies, ... while many available tools are specific to particular settings.

We propose RLStructures as a way to i) simulate multiple policies, multiple models and multiple environments simultaneously at scale ii) define complex loss functions and iii) quickly implement various policies architectures.

The main RLStructures principle is that the users delegate to the library the sampling of trajectories and episodes such that they will spend most of their time on the interesting part of RL research: developing new models and algorithms.

RLStructures is easy to use: it has very few simple interfaces that can be learned in one hour by reading the tutorials. It is not a RL Alorithms catalog, but a library to do RL Research. As illustrations, it comes with multiple RL algorithms as examples including A2C, PPO, DDQN and SAC.

## Learning RLStructures

The complete documentation is available at [http://facebookresearch.github.io/rlstructures](http://facebookresearch.github.io/rlstructures) and **tutorials** can be found in the *tutorial/* directory. The examples algorithms are provided in *raglos/*.

A facebook group is also opened for discussion.

# List of Papers using rlstructures

* [https://arxiv.org/abs/2005.02934](Learning Adaptive Exploration Strategies in Dynamic Environments Through Informed Policy Regularization)
* More to come...


# Citing RLStructures

Please use this bibtex if you want to cite this repository in your publications:

```
    @misc{rlstructures,
        author = {L. Denoyer, D. Rothermel and X. Martinet},
        title = {{RLStructures - A simple library for RL research}},
        year = {2021},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://gitHub.com/facebookresearch/rlstructures}},
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
