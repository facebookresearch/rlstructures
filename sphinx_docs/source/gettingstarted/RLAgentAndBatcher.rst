
Agents/Policies
===============

* https://github.com/facebookresearch/rlstructures/blob/main/tutorial/tutorial_rlagent.py


Methods
----------------------------------

An agent is the (only) abstraction needed to allow `rlstructures` to collect interactions at scale. One Agent corresponds to a set of policies (formally :math:`\pi_z`)

* An Agent class represents a policy (or *multiple policies* through the `agent_info` argument) acting on a **batch of environment**

* An Agent may include (or not) one or multiple pytorch modules

* The `__call__(agent_state,observation,agent_info=None,history=None)` methods take as an input:

  * `agent_state` is the state of the agent at time t-1 (as a `DictTensor`)
  * `observation` comes from the `rlstructures.VecEnv` environment
  * `agent_info` corresponds to additional (the :math:`z` in :math:`\pi_z`) information provided to the agent (e.g the value of epsilon for epsilon-greedy policies)
  * `history` may be a `TemporalDictTensor` representing a set of previous transitions (e.g. used for implementing Transformer based methods, but its value is always `None` in the default implementation of an agent), and activated only if `Agent.require_history()==True`.

* Note that `agent_state.n_elems()==observation.n_elems()` which is the number of environments on which the agent is computed.

As an output, the **__call__** method returns a pair `action,new_state` where:
  * `action` is the action outputed by the agent as a `DictTensor`. Note that `action.n_elems()==observation.n_elems()`. This information will be transmitted to the environment through the `env.step` method. Note also that the action may contain any information that you would like to store in the resulting trajectory like debugging information for instance (e.g. agent step).

  * `new_state` is the update of the state of the agent at time `t+1`. This new state is the information transmitted to the agent at the next call when acquiring a trajectory.

* `RL_Agent` implements an `initial_state(self,agent_info,B)` methods responsible of setting the initial agent state at the beginning of an episode.

Please, consider the `tutorial` examples to see different agent implementations.

Examples
--------

We provide here an example of a simple uniform RL_Agent that computes the timestep as its internal state.

.. code-block:: python

    class UniformAgent(RL_Agent):
        def __init__(self,n_actions):
            super().__init__()
            self.n_actions=n_actions

        def initial_state(self,agent_info,B):
            return DictTensor({"timestep":torch.zeros(B).long()})

        def __call__(self,state,observation,agent_info=None,history=None):
            B=observation.n_elems()

            scores=torch.randn(B,self.n_actions)
            probabilities=torch.softmax(scores,dim=1)
            actions=torch.distributions.Categorical(probabilities).sample()
            new_state=DictTensor({"timestep":state["timestep"]+1})
            return DictTensor({"action":actions}),new_state


Agent and Batcher
-----------------

An `Agent` and a `VecEnv` are used together through a `RL_Batcher` to collect trajectories.
Building a `RL_Batcher` is made as illustrated below.

First one has to define agent and environment creation methods:

.. code-block:: python

    def create_env(max_episode_steps=100,seed=None):
        envs=[]
        for k in range(4):
            e=gym.make("CartPole-v0")
            e.seed(seed)
            e=TimeLimit(e, max_episode_steps=max_episode_steps)
            envs.append(e)
        return GymEnv(envs,seed=seed)

    def create_agent(n_actions):
        return UniformAgent(n_actions)


Then the creation of the batcher is quite simple.

.. code-block:: python

    batcher=RL_Batcher(
                n_timesteps=100,
                create_agent=create_agent,
                create_env=create_env,
                agent_args={"n_actions":2},
                env_args={"max_episode_steps":100},
                n_processes=1,
                seeds=[42],
                agent_info=DictTensor({}),
                env_info=DictTensor({})
            )

* `n_timesteps` is the number of step that the batcher will acquire at each call.
* `n_processes` is the number of processes created by the batcher.
* `seeds` is a list of seed values, one per process to control the seeds of the environments in the different processes.
* `agent_info` and `env_info` are examples of information that could be sent to the Agent/Environment when acquiring trajectories. Since our current Agent and Environment don't make use of such information, we cosider empty DictTensor in our case.

With a batcher, we can use three different methods:
* batcher.reset(agent_info,env_info): It will reset both the agents and environments with the corresponding informations
* batcher.execute(agent_info=None): It will launch the acquisition of trajectories (considering agent_info, or the agent_info provided at reset if not specified)
* batcher.get: It will returns the acquired trajectories

Here is an example of use:

.. code-block:: python

    batcher.reset()
    batcher.execute()
    acquired_trajectories,n_still_running_envs=batcher.get()

* the get function returns a pair of ( `acquired trajectories` , `number of environments still running` ). Indeed, at acquisition time, some environments may stop. If no more environments are running, then one has to call `reset` again.
* the `acquired_trajectories` is a `Trajectories` object containing both an information `acquired_trajectories.info` as a DictTensor and a sequence of transitions `acquired_trajectories.trajectories` as a `TemporalDictTensor`

Trajectories returned by a batcher
----------------------------------

Let us consider `acquired_trajectories`:

* Focus on  `acquired_trajectories.info`

  * `acquired_trajectories.info.truncate_key("agent_info/")` returns the `agent_info` value used for this acquisition

  * `acquired_trajectories.info.truncate_key("env_info/")` returns the `env_info` value used for this acquisition

  * `acquired_trajectories.info.truncate_key("agent_state/")` returns the state of the agent when starting the acquisition

* Focus on `acquired_trajectories.trajectories`

  * `acquired_trajectories.trajectories["observation/"+k]` is the value of field `k` returned by the environment at time `t`

  * `acquired_trajectories.trajectories["action/"+k]` is the value of field `k` returned by the agent as action at time `t`

  * `acquired_trajectories.trajectories["_observation/"+k]` is the value of field `k` returned by the environment at time `t+1`

Note that, the final state of one episode is only available in `acquired_trajectories.trajectories["_observation/"+k]`, i.e as the `t+1` observation in the last acquired transitions
