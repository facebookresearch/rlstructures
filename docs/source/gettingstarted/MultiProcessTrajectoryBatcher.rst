# Advanced Topics - Trajectory Batcher

A trajectory batcher will just acquire N timesteps (and not full episodes)

```
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")    

```

The **EpisodeBatcher** will sample full episodes (until the environment returns done==True)
* If one consider a rlstructures.VecEnv env, and n_threads (or processes), then the batcher will sample n_episodes = N * env.n_envs()*n_threads episodes at each execution (where N is chosen by the user)
* *seeds* is a list of environment seeds, one seed per process
* The batcher has to be configured 'at the right size' since all the processes are sharing a common *Buffer* to store trajectories
* The simplest case is:
* * *n_slots = env.n_envs() x n_threads *
* * *n_timeteps* is the number of timesteps that will be acquired at each call
    
```
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

```

A trajectory batcher has to be *reset*. Then calling *execute* will acquire the next T steps. The *execute* method will return *None* if all environments have stopped
```

    batcher.reset(agent_info=DictTensor({"agent_id":torch.arange(16)}),env_info=DictTensor({"env_id":torch.arange(16)}))
    import time
    
    batcher.execute()
    t=batcher.get()
    
    while not t is None:
        batcher.execute()
        t=batcher.get()
        
```
