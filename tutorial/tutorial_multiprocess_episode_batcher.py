#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Advanced Topics

# Let us consider that we define multiple environments identified by an *environment_id*, such that two environments with two different *ids* does not behave exactly the same.
# This can be easily implemented by using the env_info* argumennt in the reset function

from rlstructures import Agent,DictTensor
import torch
import os
import sys
import gym
from gym.wrappers import TimeLimit
from gym.spaces import Discrete
from rlstructures.env_wrappers import GymEnv
from rlstructures.batchers import EpisodeBatcher,Batcher
import gym
from gym.utils import seeding

class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space=Discrete(2)

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

# As you can see, the env_info can be used as an input parameter for the environment allowing to model multiple environments through a single class

# We can do the same with agents, and implement an *Agent* that is parametrized by an *agent_info*. In our case, the agent is just an agent outputting its agent_id as an action.
# Advanced examples are shown in the *rlaglos* directory (e.g stochastic/deterministic polices, epsilon-greedy policies, ...)

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

# By specifying a particular value of *env_info* and *agent_infoo* when calling the *batcher.execute* method, the user may control which agent interacts with which environment
# Let us illustrate this using **Multi-processes batchers**

def create_env(seed=0,max_episode_steps=100):
    envs=[]
    for k in range(4):
        e=MyEnv()
        e=TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnv(envs,seed=seed)

def create_agent(n_actions=None):
    # Here, the buffer argument must be specified
    return UniformAgent(n_actions)

# Since we are using multi-process batchers, we have to switch to *spawn* mode.


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    # The **EpisodeBatcher** will sample full episodes (until the environment returns done==True)
    # If one consider a rlstructures.VecEnv env, and n_threads (or processes), then the batcher will sample n_episodes = N * env.n_envs()*n_threads episodes at each execution (where N is chosen by the user)
    # *seeds* is a list of environment seeds, one seed per process
    # The batcher has to be configured 'at the right size' since all the processes are sharing a common *Buffer* to store trajectories
    # The simplest case is to choose *n_slots=maximun  number of acquired episodes* and *n_timesteps=max size of the episodes*. We will describe later the exact meaning of theses arguments.

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

    # Execution will start the acqusition process (such that some other computations can be done in parallel to the episodes acquisition)

    n_episodes=32

    # Since we will sample 32 episodes, we need to configure the 32 agents and 32 environments that will interact

    agent_info=DictTensor({"agent_id":torch.arange(32)})
    env_info=DictTensor({"env_id":torch.arange(32)})

    #Running the batcher. It is a non-blocking function that launch the acqusition
    batcher.execute(n_episodes=32,agent_info=agent_info,env_info=env_info)

    #Getting episodes -- not that get is a blocking function such that the process will wait until the end of the acquisition.
    trajectories=batcher.get()

    #*get* can also be called in non-blocking mode *trajectories.get(blocking=Flase)* that wil return *None* if the acqusition process is not terminated
    # the **reexecute** method is a shortcut to call *execute* again with the same arguments
    batcher.reexecute()
    trajectories=batcher.get()
