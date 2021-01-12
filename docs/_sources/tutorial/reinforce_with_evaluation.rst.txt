Evaluation of RL models in other processes
==========================================

Regarding the REINFORCE implementation, one missing aspect is a good evaluation of the policy:
* the evaluation has to be done with the `deterministic` policy (while learning is made with the stochastic policy)
* the evaluation over N episodes may be long, and we would like to avoid to slow down the learning

To solve this issue, we will use another batcher in `asynchronous` mode.

Creation of the evaluation batcher
----------------------------------

The evaluation batcher can be created like the trainig batcher (but with a different number of threads and slots)

.. code-block:: python

        model=copy.deepcopy(self.learning_model)
        self.evaluation_batcher=EpisodeBatcher(
            n_timesteps=self.config["max_episode_steps"],
            n_slots=self.config["n_evaluation_episodes"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name":self.config["env_name"]
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_threads=self.config["n_evaluation_threads"],
            seeds=[self.config["env_seed"]+k*10 for k in range(self.config["n_evaluation_threads"])],
        )

Running the evaluation batcher
------------------------------

Running the evaluation batcher is made through `execute`:

.. code-block:: python

        n_episodes=self.config["n_evaluation_episodes"]
        agent_info=DictTensor({"stochastic":torch.tensor([False]).repeat(n_episodes)})
        self.evaluation_batcher.execute(n_episodes=n_episodes,agent_info=agent_info)
        self.evaluation_iteration=self.iteration

Note that we store the iteration at which the evaluation batcher has been executed

Getting trajectories without blocking the learning
--------------------------------------------------

Not we can get episodes, but in non blocking mode: the batcher will return `None` if the process of computing episodes is not finished.
If the process is finished, we can 1) compute the reward 2) update the batchers models 3) relaunch the acquisition process. We thus have an evaluation process that runs without blocking the learning, and at maximum speed.

.. code-block:: python

            evaluation_trajectories=self.evaluation_batcher.get(blocking=False)
            if not evaluation_trajectories is None: #trajectories are available
                #Compute the cumulated reward
                cumulated_reward=(evaluation_trajectories["_reward"]*evaluation_trajectories.mask()).sum(1).mean()
                self.logger.add_scalar("evaluation_reward",cumulated_reward.item(),self.evaluation_iteration)
                #We reexecute the evaluation batcher (with same value of agent_info and same number of episodes)
                self.evaluation_batcher.update(self.learning_model.state_dict())
                self.evaluation_iteration=self.iteration
                self.evaluation_batcher.reexecute()
