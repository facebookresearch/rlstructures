#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures import TemporalDictTensor, DictTensor
from .buffers import Buffer,LocalBuffer
from .threadworker import ThreadWorker
import rlstructures.logging as logging
import torch
import numpy as np
from rlstructures import DictTensor

class MultiThreadEpisodeBatcher:
    def execute(self, n_episodes, agent_info=DictTensor({}),env_info=DictTensor({})):
        n_workers = len(self.workers)
        assert n_episodes % (self.n_envs*n_workers) == 0
        assert isinstance(agent_info,DictTensor) and (agent_info.empty() or agent_info.n_elems()==n_episodes)
        assert isinstance(env_info,DictTensor) and (env_info.empty() or env_info.n_elems()==n_episodes)
        
        self.n_per_worker = [int(n_episodes / n_workers) for w in range(n_workers)]
        pos=0
        for k in range(n_workers):            
                n=self.n_per_worker[k]
                assert n%self.n_envs==0
                wi=agent_info.slice(pos,pos+n)
                ei=env_info.slice(pos,pos+n)
                self.workers[k].acquire_episodes(
                    n_episodes=self.n_per_worker[k], agent_info=wi, env_info=ei
                )
                pos+=n
        assert pos==n_episodes

    def reexecute(self):
        n_workers = len(self.workers)
        
        for k in range(n_workers):                            
                self.workers[k].acquire_episodes_again()
        
    def get(self,blocking=True):
        if not blocking:
            for w in range(len(self.workers)):
                b=self.workers[w].finished()
                if not b:    
                    return None

        max_length = 0
        buffer_slot_id_lists = []
        for w in range(len(self.workers)):
            if self.n_per_worker[w] > 0:
                sid_lists = self.workers[w].get()
                for sids in sid_lists:
                    buffer_slot_id_lists.append(sids)
                    max_length = max(max_length, len(sids))
        if max_length > 1:
            if self.warning == False:
                logging.info(
                    "================== EpisodeBatcher: trajectories over"
                    + " multiple slots => may slow down the acquisition process"
                )
                self.warning = True
            return self.buffer.get_multiple_slots(buffer_slot_id_lists, erase=True)
        else:
            buffer_slot_ids = [i[0] for i in buffer_slot_id_lists]
            return self.buffer.get_single_slots(buffer_slot_ids, erase=True)

    def update(self, info):
        for w in self.workers:
            w.update_worker(info)
       
    def close(self):
        for w in self.workers:
            w.close()


class EpisodeBatcher(MultiThreadEpisodeBatcher):
    def __init__(
        self,
        n_timesteps,
        n_slots,
        create_agent,
        agent_args,
        create_env,
        env_args,
        n_threads,
        seeds=None,
    ):
        # Buffer creation:
        agent = create_agent(**agent_args)        
        env = create_env(**{**env_args,"seed":0})
        obs,who=env.reset()
        a,b,c=agent(None,obs)
        
        self.n_envs=env.n_envs()
        specs_agent_state=a.specs()
        specs_agent_output=b.specs()
        specs_environment=obs.specs()
        del env
        del agent
        
        self.buffer = LocalBuffer(
            n_slots=n_slots,
            s_slots=n_timesteps,
            specs_agent_state=specs_agent_state,
            specs_agent_output=specs_agent_output,
            specs_environment=specs_environment,
        )
        self.workers = []
        self.n_per_worker = []
        self.warning = False

        if seeds is None:
            logging.info(
                "Seeds for batcher environments has not been chosen. Default"
                + " is None"
            )        
            seeds = [None for k in range(n_threads)]
        
        if (isinstance(seeds,int)):
            s=seeds
            seeds=[s+k*64 for k in range(n_threads)]
        assert len(seeds)==n_threads,"You have to choose one seed per thread"        
        logging.info("[EpisodeBatcher] Creating %d threads" % (n_threads))
        for k in range(n_threads):
            e_args = {**env_args, "seed": seeds[k]}
            worker = ThreadWorker(
                len(self.workers),
                create_agent,
                agent_args,
                create_env,
                e_args,
                self.buffer,
            )
            self.workers.append(worker)

    def close(self):
        super().close()
        self.buffer.close()
        

class MonoThreadEpisodeBatcher:
    def __init__(self,
        create_agent,
        agent_args,
        create_env,
        env_args,
    ):
        self.agent=create_agent(**agent_args)
        assert not self.agent.require_history()
        self.env=create_env(**env_args)
        self.n_envs=self.env.n_envs        

    def execute(self, agent_info=DictTensor({}),env_info=DictTensor({})):
        self.agent_info=agent_info
        self.env_info=env_info

    def get(self):
        with torch.no_grad():
            obs,is_running=self.env.reset(self.env_info)
            n_elems=obs.n_elems()
            observations=[{k:obs[k] for k in obs.keys()}]
            states=[]        
            agent_state=None
            agent_info=self.agent_info
            if agent_info is None:
                agent_info=DictTensor({})
            t=0
            length=torch.zeros(is_running.size()[0]).long()
            first_state=None
            first_info=agent_info
            while is_running.size()[0]>0:
                old_agent_state, agent_output, new_agent_state = self.agent(
                    agent_state, obs,agent_info
                )
                
                if (len(states)==0):
                    first_state=old_agent_state
                    s={k:old_agent_state[k] for k in old_agent_state.keys()}
                    s={**s,**{k:agent_output[k] for k in agent_output.keys()}}
                    s={**s,**{"_"+k:new_agent_state[k] for k in new_agent_state.keys()}}
                    states.append(s)
                else:
                    s={k:old_agent_state[k] for k in old_agent_state.keys()}
                    s={**s,**{k:agent_output[k] for k in agent_output.keys()}}
                    s={**s,**{"_"+k:new_agent_state[k] for k in new_agent_state.keys()}}
                                        
                    ns={k:states[0][k].clone() for k in states[0]}
                    
                    for k in states[0]:
                        ns[k][is_running]=(s[k])
                    states.append(ns)

                (l_o,l_is_running),(obs,is_running)=self.env.step(agent_output)

                for k in l_o.keys():
                    observations[t]["_"+k]=observations[0][k].clone()
                for k in l_o.keys():
                    observations[t]["_"+k][l_is_running]=(l_o[k])
                length[l_is_running]+=1
                t+=1
                if (is_running.size()[0]>0):
                    observations.append({})
                    for k in obs.keys():
                        observations[t][k]=observations[0][k].clone()
                    for k in obs.keys():
                        observations[t][k][is_running]=(obs[k])
                         
                    ag={k:first_state[k].clone() for k in first_state.keys()}                                      
                    for k in ag:
                        ag[k][l_is_running]=new_agent_state[k]
                    agent_state=DictTensor({k:ag[k][is_running] for k in ag})

                    ai={k:first_info[k].clone() for k in first_info.keys()}  
                    agent_info=DictTensor({k:ai[k][is_running] for k in ai})
                
            f_observations={}
            for k in observations[0]:
                _all=[o[k].unsqueeze(1) for o in observations]
                f_observations[k]=torch.cat(_all,dim=1)
            f_states={}
            for k in states[0]:
                _all=[o[k].unsqueeze(1) for o in states]
                f_states[k]=torch.cat(_all,dim=1)
            return TemporalDictTensor({**f_observations,**f_states},lengths=length)

        
    def update(self, info):
        self.agent.update(info)

    def close(self):
        pass
