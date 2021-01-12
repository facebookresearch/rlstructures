Environments
============

In `rlstructures`, an environment is an instance of `rlstructures.VecEnv`.

A `rlstructures.VecEnv` represents `VecEnv.n_envs()` simple environments at once.

Conceptually, a `VecEnv.step` methods takes as an input an `action` (as a `DictTensor`) and returns an obervation (as a `DictTensor`).

In practice the return type of the `VecEnv.step` returns a more complicated structure (that you don't need to understand if you don't intend to create your own environments or are usiing the openAI gym interface)

Note that typically, the observation may contain the reward obtained by the agent and any relevant information.

The `reset` function may receive an `env_info` (of size `VecEnv.n_envs()`) arguments as a dictionnary of list/np.array. It allows one to implement parametrized environments (i.e different environments in the same `VecEnv`).

Creating from Gym Environments
-------------------------------------------

The simplest way to create a `rlstructures/VecEnv` is to do it from a `gym.Env`. In that case, the `gym` env will:
* return a `dict` or a simple array as an observation
* the `observation_space` and `action_space` are not needed

Let us define a simple `gym.Env` environment:

.. code-block:: python


    import gym
    from gym.utils import seeding
    from gym.spaces import Discrete


    class MyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space=Discrete(2)

        def seed(self,seed=None):
            print("Seed = %d"%seed)
            self.np_random,seed=seeding.np_random(seed)

        def reset(self,env_info={}):
            self.x=self.np_random.rand()*2.0-1.0
            self.identifier=self.np_random.rand()
            return {"x":self.x,"identifier":self.identifier}

        def step(self,action):
            if action==0:
                self.x-=0.3
            else:
                self.x+=0.3

            return {"x":self.x,"identifier":self.identifier},self.x,self.x<-1 or self.x>1,{}

We can cast 4 environment instances to a `rlstructures.VecEnv` as follows:
.. code-block:: python

    envs=[MyEnv() for k in range(4)]
    env=GymEnv(envs,seed=80)

Each instance `i` of the `gym.Env` will be initialized with `seed+i` such that the multiple instances will have different seeds.

We also provide a wrapper allowing an infinite execution of the environment where each environment instance is automatically reseted at the end of each episode:

.. code-block:: python

    envs=[MyEnv() for k in range(4)]
    env=GymInfEnv(envs,seed=80)

To know more about rlstructures.VecEnv
--------------------------------------

1. A VecEnv corresponds to `env.n_envs()` environnements that are running simultaneously.
2. A each timestep, `n < env.n_envs()` are running since some environmens may have stopped due to end of the episode.
3. `VecEnv` returns a `DictTensor` denoted `obs` as an observation such that `obs.n_elems()==n` (i.e one observation per running environment)
3. At time `t+1`, `VecEnv.step` has to receive a `DictTensor` of size `n` (i.e one action for each running environment)

In the real-life, when executing the `reset` function, the `rlstructures.VecEnv` returns a tuple `observation,running_environments`. The `running_environment` tensor tells which environments are still running.

When executing the `step` method:

.. code-block:: python

    (obs,who_was_running),(obs2,who_is_still_running) = env.step(action)

* `obs` is the observation (at t) coming from the environments that were running at t-1
* `who_was_running` is the list of environnments still running at time t-1. Note that `who_was_running.size()[0]=obs.n_elems()`
* `obs2` is the observation (at t) from the environments that are still running at time t (i.e `obs2` is a subset of `obs`)
* `who_is_still_running` is the list of environments running at time t


Interacting with the Environment
--------------------------------

Interaction with the environment is easy, the agent and environment exchanging `DictTensor`

.. code-block:: python
    obs,who_is_still_running=env.reset()
    print(obs)
    n_running=who_is_still_running.size()[0]
    while n_running>0: #While some envs are still running
        action=DictTensor({"action":torch.tensor([0]).repeat(n_running)})
        (obs,who_was_running),(obs2,who_is_still_running) = env.step(action)
        n_running=who_is_still_running.size()[0]
        print(obs2)
