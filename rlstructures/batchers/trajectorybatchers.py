#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures import TemporalDictTensor,  DictTensor
from .buffers import Buffer,LocalBuffer
from .threadworker import ThreadWorker
import rlstructures.logging as logging
import torch
import numpy as np


class MultiThreadTrajectoryBatcher:
    def reset(self,agent_info=DictTensor({}), env_info=DictTensor({})):
        n_workers = len(self.workers) 
        assert isinstance(agent_info,DictTensor) and (agent_info.empty() or agent_info.n_elems()==self.n_envs*n_workers)
        assert isinstance(env_info,DictTensor) and (env_info.empty() or env_info.n_elems()==self.n_envs*n_workers)                
        pos=0
        for k in range(n_workers):
                n=self.n_envs
                wi=None if agent_info is None else agent_info.slice(pos,pos+n)
                ei= None if env_info is None else env_info.slice(pos,pos+n)
                self.workers[k].reset(
                    agent_info=wi, env_info=ei
                )
                pos+=n                

    def execute(self):
        n_workers = len(self.workers)         
        for k in range(n_workers):                
                self.workers[k].acquire_slot()

    def get(self,blocking=True):
        if not blocking:
            for w in range(len(self.workers)):
                if not self.workers[w].finished():
                    return None

        buffer_slot_ids = []
        for w in range(len(self.workers)):
            buffer_slot_ids += self.workers[w].get()
        if len(buffer_slot_ids)==0: return None
        slots = self.buffer.get_single_slots(buffer_slot_ids, erase=True)
        assert not slots.lengths.eq(0).any()
        return slots

    def update(self, info):
        for w in self.workers:
            w.update_worker(info)
        
    def close(self):
        for w in self.workers:
            w.close()
        for w in self.workers:
            del w



class Batcher(MultiThreadTrajectoryBatcher):
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

        if seeds is None:
            logging.info(
                "Seeds for batcher environments has not been chosen. Default is None"
            )
            seeds = [None for k in range(n_threads)]
        
        if (isinstance(seeds,int)):
            s=seeds
            seeds=[s+k*64 for k in range(n_threads)]
        assert len(seeds)==n_threads,"You have to choose one seed per thread"

        logging.info("[Batcher] Creating %d threads " % (n_threads))
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