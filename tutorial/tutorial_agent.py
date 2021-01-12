
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Agents/Policies in RLStructures
# 
# An agent is the (only) abstraction need to allow RLStructures to collect interactions at scale. 
# * An Agent class represents a policy (or *multiple policies*)
# * An Agent may include (or not) one or multiple pytorch Module, but it is not mandatory
# * An Agent is stateless, and only implements a **__call__** methods
# * The **__call__(agent_state,observation,agent_info)** methods takes as an input:
# * * The state of the agent as time t-1 (as a DictTensor). The state will store all the informations needed to continue the execution of the agent
# * * The observation provided by the *rlstructures.VecEnv* environments
# * * * Note that agent_state.n_elems()==observation.n_elems()
# * * an *agent_info* DictTensor that is an additional information that will be controlled by the user
# * * * For instace, *agent_info* will contains a flag telling if the agent has to be stochastic or deterministic

# As an output, the **__call__** method returns *(old_state,action,new_state)* where:
# * *action* is the action outputed by the agent as a DictTensor. Note that *action.n_elems()==observation.n_elems()*. This information will be transmitted to the environment through the *env.step* method.
# * *new_state* is the update of the state of the agent. This new state is the information transmitted to the Agent at the next call of the agent
# * *old_state* is the state of the agent before action/new_state computation
# * * In most of the cases, *old_state* is strictly equal to *agent_state*
# * * When *agent_state is None*, then *old_state* will initialize the state of the agent at *t==0*
# * * In some other cases (e.g reinitialization of the state of the agent during one episode), having *agent_state!=old_state* may be very useful to implement complex agents.


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


# Agent and Batcher
# 
# An *Agent* and a *VecEnv* are used together into a **Batcher** to collect episodes or trjaectories (a trajectory is a piece of episode)
# The simplest Batcher is the **MonoThreadEpisodeBatcher** which is running in the main process. Other batcher are in RLStructures:
# * The *EpisodeBatcher* is a multi-process batcher sampling full episodes
# * The *Batcher* is a multi-process batcher sampling N timesteps 
# The complex batchers are explained later

# For creating a batcher, one has to provide **(pickable) functions and arguments** and not built object. Indeed, the batchers are taking in charge the creation of the objects.

import gym
from gym.wrappers import TimeLimit
from rlstructures.env_wrappers import GymEnv

def create_env(max_episode_steps=100):
    envs=[]
    for k in range(4):
        e=gym.make("CartPole-v0")
        e=TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnv(envs,seed=10)

def create_agent(n_actions):
    return UniformAgent(n_actions)


# The creation of the batcher is simple


from rlstructures.batchers import MonoThreadEpisodeBatcher
batcher=MonoThreadEpisodeBatcher(
        create_agent=create_agent,
        agent_args={"n_actions":2},
        create_env=create_env,
        env_args={"max_episode_steps":100}
)

# Depending on the batcher, one may then use different acquisition functions
# In the mono-process case, on can use the 
# * * *execute(agent_info,env_info)* function returns env.n_envs() episodes
# * * Acquired episodes are accessible by calling the *get* method that returns a *TemporalDictTensor*

batcher.execute()
trajectories=batcher.get()
print("Lengths of trajectories = ",trajectories.lengths)



