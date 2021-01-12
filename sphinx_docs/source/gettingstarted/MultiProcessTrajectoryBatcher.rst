Multiprocess Batcher
====================

A trajectory batcher will just acquire N timesteps (and not full episodes)

* `n_timesteps` is the number of steps to acquire at each call
* `n_slots` is the number of simulatenous acquisition which is typically `n_slots=n_threads*n_envs`

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
