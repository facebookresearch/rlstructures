rlstructures
============

TL;DR
-----
`rlstructures` is a lightweight Python library that provides simple APIs as well as data structures that make as few assumptions as possible on the structure of your agent or your task while allowing the transparent execution of multiple policies on multiple environments in parallel (incl. multiple GPUs).

Important Note (Feb 2021) -- version 0.2
----------------------------------------

Due to feedback, we have made changes over the API (v0.2)

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

* A growing series of tutorials at [https://medium.com/@ludovic.den](https://medium.com/@ludovic.den)

All these changes are documented in the HTML documentation at [http://facebookresearch.github.io/rlstructures](http://facebookresearch.github.io/rlstructures)

Why/What?
---------
RL research addresses multiple aspects of RL like hierarchical policies, option-based policies, goal-oriented policies, structured input/output spaces, transformers-based policies, etc., and there are currently few tools to handle this diversity of research projects.

We propose `rlstructures` as a way to:

* Simulate multiple policies, multiple models and multiple environments simultaneously at scale

* Define complex loss functions

* Quickly implement various policy architectures.

The main RLStructures principle is that the users delegates the sampling of trajectories and episodes to the library so they can spend most of their time on the interesting part of RL research: developing new models and algorithms.

`rlstructures` is easy to use: it has very few simple interfaces that can be learned in one hour by reading the tutorials.

It comes with multiple RL algorithms as examples including A2C, PPO, DDQN and SAC.

Please reach out to us if you intend to use it. We will be happy to help, and potentially to implement missing functionalities.

Targeted users
--------------

RLStructures comes with a set of implemented RL algorithms. But rlstructures does not aim at being a repository of benchmarked RL algorithms (an other RL librairies do that very well). If your objective is to apply state-of-the-art methods on particular environments, then rlstructures is not the best fit. If your objective is to implement new algorithms, then rlstructures is a good fit.

Where?
------

* Github: http://github.com/facebookresearch/rlstructures
* Tutorials: https://medium.com/@ludovic.den
* Discussion Group: https://www.facebook.com/groups/834804787067021

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   overview
   gettingstarted/index
   algorithms/index
   api/index
   foireaq/foireaq.rst
   migrating_v0.1_v0.2
   deprecated/index.rst
