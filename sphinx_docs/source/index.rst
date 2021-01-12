rlstructures
============

TL;DR
-----
`rlstructures` is a lightweight Python library that provides simple APIs as well as data structures that make as few assumptions as possible on the structure of your agent or your task while allowing the transparent execution of multiple policies on multiple environments in parallel (incl. multiple GPUs).

Why/What?
---------
RL research addresses multiple aspects of RL like hierarchical policies, option-based policies, goal-oriented policies, structured input/output spaces, transformers-based policies, etc., and there are currently few tools to handle this diversity of research projects.

We propose `rlstructures` as a way to:

* Simulate multiple policies, multiple models and multiple environments simultaneously at scale

* Define complex loss functions

* Quickly implement various policy architectures.

The main RLStructures principle is that the users delegates the sampling of trajectories and episodes to the library so they can spend most of their time on the interesting part of RL research: developing new models and algorithms.

`rlstructures` is easy to use: it has very few simple interfaces that can be learned in one hour by reading the tutorials. It comes with multiple RL algorithms as examples including A2C, PPO, DDQN and SAC. Besides, there are 5 projects already using it (Multitask RL, Exploration, Diversity in RL, Optimization, ...) and helping each other.
Please reach out to us if you intend to use it. We will be happy to help, and potentially to implement missing functionalities.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   overview
   gettingstarted/index
   tutorial/index
   algorithms/index
   api/index
   foireaq/foireaq.rst
