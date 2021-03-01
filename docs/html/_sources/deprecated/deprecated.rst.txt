
Agents/Policies
===============

**The Batcher, Agent, EpisodeBatcher classes are now deprecated (but still working). The simplified alternative is to use RL_Agent and RL_Batcher instead.**

* https://github.com/facebookresearch/rlstructures/blob/main/tutorial/tutorial_agent.py


An agent is a parameterized policy
----------------------------------

An agent is the (only) abstraction needed to allow `rlstructures` to collect interactions at scale. One Agent corresponds to a set of policies (formally :math:`\pi_z`)

* An Agent class represents a policy (or *multiple policies* through the `agent_info` argument) acting on a **batch of environment**

* An Agent may include (or not) one or multiple pytorch modules

* An Agent is stateless, and only implements a `__call__` method

* The `__call__(agent_state,observation,agent_info=None,history=None)` methods take as an input:

  * `agent_state` is the state of the agent at time t-1 (as a `DictTensor`)
  * `observation` comes from the `rlstructures.VecEnv` environment
  * `agent_info` corresponds to additional (the :math:`z` in :math:`\pi_z`) information provided to the agent (e.g the value of epsilon for epsilon-greedy policies)
  * `history` may be a `TemporalDictTensor` representing a set of previous transitions (e.g. used for implementing Transformer based methods, but its value is always `None` in the default implementation of an agent), and activated only if `Agent.require_history()==True`.

* Note that `agent_state.n_elems()==observation.n_elems()` which is the number of environments on which the agent is computed.
* `agent_info=None, history=None` is mandatory in the method definition, and the agent must initialize, for itself, the value of `agent_info` if `agent_info is None`

As an output, the **__call__** method returns a triplet `(old_state,action,new_state)` where:

* `action` is the action outputed by the agent as a `DictTensor`. Note that `action.n_elems()==observation.n_elems()`. This information will be transmitted to the environment through the `env.step` method. Note also that the action may contain any information that you would like to store in the resulting trajectory like debugging information for instance (e.g. agent step).

* `new_state` is the update of the state of the agent at time `t+1`. This new state is the information transmitted to the agent at the next call.

* `old_state` is the state of the agent before action/new_state computation

  * Conceptually, `(old_state,observation,action,new_state)` corresponds to a transition in the underlying MDP

  * In most of the cases, `old_state` is strictly equal to `agent_state`

  * When `agent_state is None`, the agent will have to initialize itself, such that `old_state` will be the initial state of the agent

Please, consider the `tutorial` examples to see different agent implementations.

We provide here an example of Agent:

.. code-block:: python

    from rlstructures import Agent,DictTensor
    import torch

    class UniformAgent(Agent):
        def __init__(self,n_actions):
            super().__init__()
            self.n_actions=n_actions

        def __call__(self,state,observation,agent_info=None,history=None):
            B=observation.n_elems()

            agent_state=None
            if state is None:
                agent_state=DictTensor({"timestep":torch.zeros(B).long()})
            else:
                agent_state=state

            scores=torch.randn(B,self.n_actions)
            probabilities=torch.softmax(scores,dim=1)
            actions=torch.distributions.Categorical(probabilities).sample()
            new_state=DictTensor({"timestep":agent_state["timestep"]+1})
            return agent_state,DictTensor({"action":actions}),new_state


Agent and Batcher
-----------------

An `Agent` and a `VecEnv` are used together through a `Batcher` to collect episodes or trajectories (a trajectory is a piece of episode). The simplest Batcher is the `MonoThreadEpisodeBatcher` which is running in the main process.
`rlstructures` also provides:
* `EpisodeBatcher` which is a multi-processes batcher sampling full episodes
* `Batcher` which is a multi-processed batcher sampling N next timesteps

The multi-process batchers are described later in the documentation.

Creating a batcher involved providing functions able to create the 'rlstructures.VecEnv' and the 'rlstructures.Agent' objects as `(pickable) functions and arguments`.

.. code-block:: python

    import gym
    from gym.wrappers import TimeLimit
    from rlstructures.env_wrappers import GymEnv

    def create_env(max_episode_steps=100):
        envs=[]
        for k in range(4):
            e=gym.make("CartPole-v0")
            e=TimeLimit(e, max_episode_steps=max_episode_steps)
        return GymEnv(envs,seed=10)

    def create_agent(n_actions):
        return UniformAgent(n_actions)


The creation of the batcher is quite simple.

.. code-block:: python

    from rlstructures.batchers import EpisodeBatcher
    batcher=EpisodeBatcher(
            create_agent=create_agent,
            agent_args={"n_actions":2},
            create_env=create_env,
            env_args={"max_episode_steps":100}
    )

Depending on the batcher, one may then use different acquisition functions
In the mono-process case, one can use the

* `execute(agent_info=None,env_info=None)` function returns env.n_envs() episodes
* Acquired episodes are accessible by calling the *get* method that returns a *TemporalDictTensor*
* Note that, at each execution, the user has to provide an `agent_info` value that will be transmitted to each of the agents, and an `env_info` value that will be transmitted to each environment (through the `reset` function), allowing the execution of multiple policies on multiple environments in a single batcher call.

.. code-block:: python
    batcher.execute()
    trajectories=batcher.get()
    print("Lengths of trajectories = ",trajectories.lengths)

And finally, consider that each agent implements the `Agent.update` function that will allow one to update the parameters of the agent.

Batchers
========

Batchers are objects allowing the execution of multiple policies over multiple environments, using multiple processes. We provide two batchers:
1) `EpisodeBatcher` to acquire complete episodes (until a `done` is reached)
2) `Batcher` to acquire the `n` next timesteps over multiple environments

Examples about how to use batchers are given in the other sections.

**Important**: All the information (e.g `DictTensor`) produced by both the `Agent` and the `Env` will be available to the user in the returned `TemporalDictTensor`

Parallelization Schema
----------------------

The generic parallelization schema is illustrated in the following picture.

.. image:: https://raw.githubusercontent.com/facebookresearch/rlstructures/main/docs/images/batchers.jpg?token=ABNXVXPVRMSMY5XGYBMOUILAA725Q
  :width: 1024
  :alt: Parallelization Schema

* One batcher creates multiple processes

* Each process contains a copy of the `Agent` and a copy of the `rlstructures.VecEnv` (the copy are made through `create_agent` and `create_env` functions that are arguments at the batcher creation)

  * In our case, each Agent as its own copy of the pytorch model (note that it can be a shared_memory model to avoid to use extra memory)

* Each `VecEnv` corresponds to multiple simple environments

* At `execute`, the processes start to acquire information by simulating the interaction betweeh the agent and the environments

* At `get`, the information collected by the processes are merged to a `TemporalDictTensor` that is the output of `get`

* The call of `Batcher.update` will call `Agent.update` in all the processes to typically update the model of each Agent

Multiprocess Batcher
====================

* https://github.com/facebookresearch/rlstructures/blob/main/tutorial/tutorial_multiprocess_trajectory_batcher.py

A trajectory batcher will just acquire N timesteps (and not full episodes)

* `n_timesteps` is the number of steps to acquire at each call
* `n_slots` is the number of simulatenous acquisitions which is typically `n_slots=n_threads*n_envs`

.. code-block:: python

    batcher=Batcher(
            n_timesteps=100,
            n_slots=16,
            n_threads=4,
            seeds=[1,2,3,4],
            create_agent=create_agent,
            agent_args={"n_actions":2},
            create_env=create_env,
            env_args={"max_episode_steps":100}
    )

A trajectory batcher has to be `reset` with corresponding `agent_info` and `env_info` values.

Then calling `execute` will acquire the next T steps (over environment instances that are still running).
The `execute` method will return `None` if all environments have stopped

.. code-block:: python

    batcher.reset(agent_info=DictTensor({"agent_id":torch.arange(16)}),env_info=DictTensor({"env_id":torch.arange(16)}))
    import time

    batcher.execute()
    t=batcher.get()

    while not t is None:
        batcher.execute()
        t=batcher.get(blocking=True)

Multiprocess Episode Batcher
============================

* https://github.com/facebookresearch/rlstructures/blob/main/tutorial/tutorial_multiprocess_episode_batcher.py

Let us consider that we define multiple environments identified by an *environment_id*, such that two environments with two different *ids* does not behave exactly the same. This can be easily implemented by using the env_info* argument in the reset function:

.. code-block:: python

    from rlstructures import Agent,DictTensor
    import torch
    import os
    import sys
    import gym
    from gym.wrappers import TimeLimit
    from rlstructures.env_wrappers import GymEnv
    from rlstructures.batchers import EpisodeBatcher,Batcher
    import gym
    from gym.utils import seeding

    class MyEnv(gym.Env):
        def __init__(self):
            super().__init__()

        def seed(self,seed=None):
            self.np_random,seed=seeding.np_random(seed)

        def reset(self,env_info={"env_id":0}):
            assert "env_id" in env_info
            self.env_id=env_info["env_id"]
            self.x=self.np_random.rand()*2.0-1.0
            self.identifier=self.np_random.rand()
            obs={"x":self.x,"identifier":self.identifier,"env_id":self.env_id}
            return obs

        def step(self,action):
            if action==0:
                self.x-=0.3
            else:
                self.x+=0.3
            done = self.x<-1 or self.x>1

            obs={"x":self.x,"identifier":self.identifier,"env_id":self.env_id},self.x,done,{}
            return obs

As you can see, the env_info can be used as an input parameter for the environment allowing one to model multiple environments through a single class.

We can do the same with agents, and implement an *Agent* that is parametrized by an *agent_info*. In our case, the agent is just an agent outputting its agent_id as an action. Advanced examples are shown in the *rlaglos* directory (e.g stochastic/deterministic polices, epsilon-greedy policies, ...)

.. code-block:: python

    class UniformAgent(Agent):
        def __init__(self,n_actions):
            super().__init__()
            self.n_actions=n_actions

        def __call__(self,state,observation,agent_info=None,history=None):
            B=observation.n_elems()
            agent_state=None

            #Initialize agent_info is not specified
            if agent_info is None:
                agent_info=DictTensor({"agent_id":torch.tensor([0]).repeat(B)})

            #initialize the state of the agent if not specified
            if state is None:
                agent_state=DictTensor({"timestep":torch.zeros(B).long()})
            else:
                agent_state=state

            scores=torch.randn(B,self.n_actions)
            probabilities=torch.softmax(scores,dim=1)
            actions=torch.distributions.Categorical(probabilities).sample()
            new_state=DictTensor({"timestep":agent_state["timestep"]+1})
            # We also decide to output the action probabilities
            return agent_state,DictTensor({"action":actions,"action_probabilities":probabilities,"agent_id":agent_info["agent_id"]}),new_state

By specifying a particular value of `env_info` and `agent_info` when calling the `batcher.execute` method, the user may control which agent interacts with which environment.
Let us illustrate this using **Multi-processes batchers**

.. code-block:: python


    def create_env(seed=0,max_episode_steps=100):
        envs=[]
        for k in range(4):
            e=MyEnv()
            e=TimeLimit(e, max_episode_steps=max_episode_steps)
            envs.append(e)
        return GymEnv(envs,seed=seed)

    def create_agent(buffer=None,n_actions=None):
        # Here, the buffer argument must be specified
        return UniformAgent(buffer,n_actions)

Since we are using multi-process batchers, we have to switch to *spawn* mode.

.. code-block:: python

    if __name__ == "__main__":
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")


The **EpisodeBatcher** will sample full episodes (until the environment returns `done==True`)
If one considers a `rlstructures.VecEnv` env, and `n_threads` (or processes), then the batcher will sample `n_episodes = N * env.n_envs()*n_threads` episodes at each execution (where `N` is chosen by the user)
* `seeds` is a list of environment seeds, one seed per process
* `n_timesteps` is the maximum size of the episodes
* `n_slots` is the maximum number of episodes simultaneously acquired

.. code-block:: python

    batcher=EpisodeBatcher(
            n_timesteps=100,
            n_slots=128,
            n_threads=4,
            seeds=[1,2,3,4],
            create_agent=create_agent,
            agent_args={"n_actions":2},
            create_env=create_env,
            env_args={"max_episode_steps":100}
    )


 Since we will sample 32 episodes, we need to configure the 32 agents and 32 environments that will interact:

.. code-block:: python

    agent_info=DictTensor({"agent_id":torch.arange(32)})
    env_info=DictTensor({"env_id":torch.arange(32)})


Executing the batcher will start the acquisition process. It is a non-blocking function that launches the acqusition:

.. code-block:: python

    batcher.execute(n_episodes=32,agent_info=agent_info,env_info=env_info)

Getting episodes is done by using `get`. Note that when `blocking=True`, the process will wait until the end of the acquisition (examples of `blocking=False` are given in the `tutorials`).

.. code-block:: python

    trajectories=batcher.get(blocking=True)

* the `reexecute` method is a shortcut to call `execute` again with the same arguments:

.. code-block:: python

    batcher.reexecute()
    trajectories=batcher.get()
