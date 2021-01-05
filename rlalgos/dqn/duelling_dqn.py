#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlalgos import BaseExperiment
from rlstructures.logger import Logger, TFLogger
from rlstructures import DictTensor, TemporalDictTensor
from rlstructures import logging
from rlstructures.tools import weight_init
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
import pickle
import numpy as np
from rlstructures.batchers import Batcher,EpisodeBatcher


"""
Implementation on the DQN Alorithms (Dueling Q + Advantage). The principle is:
* we store transitions in a replay buffer. Since the algorithm can deal with POMDP, a transition is actually a piece of trajectory of at most "replay_buffer_ntimesteps". 
These transitions are sampled by: executing agent over "learning_timesteps", then collection all possible transitions of at most "replay_buffer_ntimesteps"

0) Sample initial trajectories for filling the replay buffer
1) At each iterations,
    - sample a new ste of transitions
    - compute qvalue_epochs on the DQN loss

Evaluation is made over other multiple threads (on CPUs) continously (with epsilon = 0.0)
"""

class ReplayBuffer:
    '''
    This class is used to store transitions. Each transition is a TemporalDictTensor of size T
    '''
    def __init__(self,N):
        self.N=N        
        self.buffer=None
    
    def _init_buffer(self,trajectories):
        self.buffer={}
        for k in trajectories.keys():
            dtype=trajectories[k].dtype
            size=trajectories[k].size()
            b_size=(self.N,)+size[2:]
            self.buffer[k]=torch.zeros(*b_size,dtype=dtype)
        self.pos=0
        self.full=False

    def write(self,trajectories):
        rs={}
        new_pos=None
        for k in trajectories.keys():
            v=trajectories[k]
            size=v.size()
            b_size=(size[0]*size[1],)+size[2:]
            v=v.reshape(*b_size)
            n=v.size()[0]
            overhead=self.N-(self.pos+n)            
            if new_pos is None:
                new_pos=torch.arange(n)+self.pos
                mask=new_pos.ge(self.N).float()
                nidx=torch.arange(n)+self.pos-self.N
                new_pos=(new_pos*(1-mask)+mask*(nidx)).long()
                
            self.buffer[k][new_pos]=v
        self.pos=self.pos+n
        if self.pos>=self.N:
            self.pos=self.pos-self.N
            self.full=True
        assert self.pos<self.N

    def size(self):
        if self.full:
            return self.N
        else:
            return self.pos


    def push(self,trajectories):
        '''
        Add transitions to the replay buffer
        '''
        max_length=trajectories.lengths.max().item()
        assert trajectories.lengths.eq(max_length).all()        
        if self.buffer is None:
            self._init_buffer(trajectories)
        self.write(trajectories)
        
    def sample(self,n=1):
        limit=self.pos
        if self.full:
            limit=self.N
        transitions=torch.randint(0,high=limit,size=(n,))
        d={k:self.buffer[k][transitions] for k in self.buffer}
        return DictTensor(d)

class DQN(BaseExperiment):
    def __init__(self, config, create_env, create_agent):
        super().__init__(config,create_env,create_agent)
        env = self._create_env(
            self.config["n_envs"], seed=0,**{k:self.config[k] for k in self.config if k.startswith("environment/")}
        )
        self.n_actions = env.action_space.n
        self.obs_dim = env.reset()[0]["frame"].size()[1]        
        del env

    def check_arguments(self,args):
        assert args["n_evaluation_rollouts"]%(args["n_envs"]*args["n_evaluation_threads"])==0
        return True
        
       

    def reset(self):
        self.target_model=copy.deepcopy(self.learning_model)

        model=copy.deepcopy(self.learning_model)
        
        self.train_batcher=Batcher(
            n_timesteps=self.config["batch_timesteps"],
            n_slots=self.config["n_envs"]*self.config["n_threads"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "mode":"train",
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                **{k:self.config[k] for k in self.config if k.startswith("environment/")}
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_threads=self.config["n_threads"],
            seeds=self.config["env_seed"],            
        )
        
        model=copy.deepcopy(self.learning_model)
        self.evaluation_batcher=EpisodeBatcher( 
            n_timesteps=self.config["max_episode_steps"],
            n_slots=self.config["n_evaluation_rollouts"],   
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={        
                "mode":"evaluation",                      
                "max_episode_steps": self.config["max_episode_steps"],
                "n_envs": self.config["n_envs"],
                **{k:self.config[k] for k in self.config if k.startswith("environment/")}

            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_threads=self.config["n_evaluation_threads"],
            seeds=self.config["env_seed"]*10,
        )

        self.register_batcher(self.train_batcher)
        self.register_batcher(self.evaluation_batcher)
        

    def _state_dict(self,model,device):
        sd = model.state_dict()
        for k, v in sd.items():
            sd[k] = v.to(device)
        return sd

    def run(self):
        self.replay_buffer=ReplayBuffer(self.config["replay_buffer_size"])
        device = torch.device(self.config["learner_device"])
        self.learning_model.to(device)
        self.target_model.to(device)
        optimizer = torch.optim.Adam(
            self.learning_model.parameters(), lr=self.config["lr"]
        )
        
        self.train_batcher.update(self._state_dict(self.learning_model,torch.device("cpu")))
        self.evaluation_batcher.update(self._state_dict(self.learning_model,torch.device("cpu")))

        n_episodes=self.config["n_envs"]*self.config["n_threads"]
        self.train_batcher.reset(agent_info=DictTensor({"epsilon":torch.ones(n_episodes)*self.config["epsilon_greedy"]}))
        logging.info("Sampling initial transitions")
        for k in range(self.config["initial_buffer_epochs"]):
            self.train_batcher.execute()        
            trajectories=self.train_batcher.get()
            self.replay_buffer.push(trajectories)
        
        n_episodes=self.config["n_evaluation_rollouts"]
        self.evaluation_batcher.execute(agent_info=DictTensor({"epsilon":torch.zeros(n_episodes)}), n_episodes=n_episodes)
        
        logging.info("Starting Learning")
        _start_time=time.time()
        
        logging.info("Learning")
        while time.time()-_start_time <self.config["time_limit"]:
            
            self.train_batcher.execute()
            trajectories=self.train_batcher.get()
            self.replay_buffer.push(trajectories)
            self.logger.add_scalar("replay_buffer_size",self.replay_buffer.size(),self.iteration)
            # avg_reward = 0
           
            for k in range(self.config["qvalue_epochs"]):
                optimizer.zero_grad()
                dt = self.get_loss(device)
                
                [self.logger.add_scalar(k,dt[k].item(),self.iteration) for k in dt.keys()]
                
                floss=dt["q_loss"]
                floss.backward()
                if self.config["clip_grad"] > 0:
                    n = torch.nn.utils.clip_grad_norm_(
                        self.learning_model.parameters(), self.config["clip_grad"]
                    )
                    self.logger.add_scalar("grad_norm", n.item(), self.iteration)
                self.iteration+=1
                optimizer.step()
            
                tau=self.config["tau"]
                self.soft_update_params(self.learning_model,self.target_model,tau)
                


            self.train_batcher.update(self._state_dict(self.learning_model,torch.device("cpu")))            
            self.evaluate()
            self.iteration+=1

    def soft_update_params(self,net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +(1 - tau) * target_param.data)


    def evaluate(self,relaunch=True):        
        evaluation_trajectories = self.evaluation_batcher.get(blocking=False)
        
        if (evaluation_trajectories is None):
            return
        
        avg_reward = (
                    (
                        evaluation_trajectories["_reward"]
                        * evaluation_trajectories.mask()
                    )
                    .sum(1)
                    .mean()
                    .item()
        )
        self.logger.add_scalar("avg_reward", avg_reward, self.iteration)
        if (self.config["verbose"]):
                print("Iteration "+str(self.iteration)+", Reward =  "+str(avg_reward)+", Buffer size = "+str(self.replay_buffer.size()))
        
        if (relaunch):
            self.evaluation_batcher.update(self._state_dict(self.learning_model,torch.device("cpu")))
            self.evaluation_batcher.reexecute()
        return avg_reward



    def get_loss(self, device):
        transitions=self.replay_buffer.sample(n=self.config["n_batches"])
        transitions = transitions.to(device)
        B=transitions.n_elems()
        Bv=torch.arange(B)
        action = transitions["action"]
        reward = transitions["_reward"]
        frame = transitions["frame"]
        _frame = transitions["_frame"]
        _done = transitions["_done"].float()

        q=self.learning_model(frame)
        qa=q[Bv,action]
        qp = self.learning_model(_frame)
        actionp=qp.max(1)[1]
        _q_target = self.target_model(_frame).detach()
        _q_target_a= _q_target[Bv,actionp]
        _target_value=_q_target_a*(1-_done)*self.config["discount_factor"]+reward
        td = (_target_value-qa)**2
        dt = DictTensor(
            {
                "q_loss": td.mean(),
            }
        )
        return dt

