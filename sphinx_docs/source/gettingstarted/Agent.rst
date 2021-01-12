
Agents/Policies
===============

An agent is a parameterized policy
----------------------------------

An agent is the (only) abstraction needed to allow `rlstructures` to collect interactions at scale. One Agent corresponds to a set of policy (formally :math:`\pi_z`)

* An Agent class represents a policy (or *multiple policies* through the `agent_info` argument) acting on a **batch of environment**

* An Agent may include (or not) one or multiple pytorch modules

* An Agent is stateless, and only implements a `__call__` method

* The `__call__(agent_state,observation,agent_info=None,history=None)` methods takes as an input:

  * `agent_state` is the state of the agent as time t-1 (as a `DictTensor`)
  * `observation` comes from the `rlstructures.VecEnv` environment
  * `agent_info` corresponds additional (the :math:`z` in :math:`\pi_z`) information provided to the agent (e.g the value of epsilon for epsilon-greedy policies)
  * `history` may be a `TemporalDictTensor` representing a set of previous transitions (e.d used for implementing Transformer based methods, but value is always `None` in the default implementation of an agent.), and activated only if `Agent.require_history()==True`.

* Note that `agent_state.n_elems()==observation.n_elems()` which is the number of environments on which the agent is computed.
* `agent_info=None, history=None` is mandatory in the method definition, and the agent must initialize itself the value of `agent_info` if `agent_info is None`

As an output, the **__call__** method returns a triplet `(old_state,action,new_state)` where:

* `action` is the action outputed by the agent as a `DictTensor`. Note that `action.n_elems()==observation.n_elems()`. This information will be transmitted to the environment through the `env.step` method. Note also that the action may contain any information that you would like to store in the resulting trajectory like debugging information for instance (e.g agent step).

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
* Note that, at each execute, the user as to provide an `agent_info` value that will be transmitted to each of the agents, and an `env_info` value that will be transmitted to each environment (through the `reset` function), allowing to execute multiple policies on multiple environments in a single batcher call.

.. code-block:: python
    batcher.execute()
    trajectories=batcher.get()
    print("Lengths of trajectories = ",trajectories.lengths)

At last, consider that each agent implements the `Agent.update` function that will allow one to update the parameters of the agent.
