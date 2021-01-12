# TL;DR

TL;DR: RLStructures is a lightweight Python library that provides simple APIs as well as data structures that make as few assumptions as possible about the structure of your agent or your task, while allowing for transparently executing multiple policies on multiple environments in parallel (incl. multiple GPUs). It thus facilitates the implementation of RL algorithms while avoiding complex abstractions.

# Why/What?

RL research addresses multiple aspects of RL like hierarchical policies, option-based policies, goal-oriented policies, structured input/output spaces, transformer-based policies, etc. while many available tools are specific to particular settings.

We propose RLStructures as a way to i) simulate multiple policies, multiple models and multiple environments simultaneously at scale ii) define complex loss functions and iii) quickly implement various policy architectures.

The main principle of RLStructures is to allow the user to delegate the sampling of trajectories and episodes to the library so they can spend most of their time on the interesting part of RL research: developing new models and algorithms.

RLStructures is easy to use: it has very few simple interfaces that can be learned in one hour by reading the tutorials. It is not a RL Alorithms catalog, but a library to do RL Research. For illustration purposes it comes with multiple RL algorithms including A2C, PPO, DDQN and SAC.

## Learning RLStructures

* [http://facebookresearch.github.io/rlstructures](http://facebookresearch.github.io/rlstructures)

The complete documentation is available at [http://facebookresearch.github.io/rlstructures](http://facebookresearch.github.io/rlstructures) and **tutorials** can be found in the *tutorial/* directory. The example algorithms are provided in *raglos/*.

A facebook group is also open for discussion.

# List of Papers using rlstructures

* [Learning Adaptive Exploration Strategies in Dynamic Environments Through Informed Policy Regularization](https://arxiv.org/abs/2005.02934)
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

## License

`rlstructures` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
