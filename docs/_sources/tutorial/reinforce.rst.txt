Implementing Reinforce with rlstructures
========================================

We explain how we can quickly implement a REINFORCE algorithm working on multiple processes (see `tutorial/tutorial_reinforce`). Note that all the provided algorithms produce a tensorboard and CSV output.

Creating the policy
-------------------

The first step is to create the pytorch model for both the policy and the baseline (in 'agent.py'). We will use simple MLP with one hidden layer.

.. code-block:: python

    class AgentModel(nn.Module):
        """ The model that computes one score per action
        """
        def __init__(self, n_observations, n_actions, n_hidden):
            super().__init__()
            self.linear = nn.Linear(n_observations, n_hidden)
            self.linear2 = nn.Linear(n_hidden, n_actions)


        def forward(self, frame):
            z = torch.tanh(self.linear(frame))
            score_actions = self.linear2(z)
            probabilities_actions = torch.softmax(score_actions,dim=-1)
            return probabilities_actions

    class BaselineModel(nn.Module):
        """ The model that computes V(s)
        """
        def __init__(self, n_observations, n_hidden):
            super().__init__()
            self.linear = nn.Linear(n_observations, n_hidden)
            self.linear2 = nn.Linear(n_hidden, 1)


        def forward(self, frame):
            z = torch.tanh(self.linear(frame))
            critic = self.linear2(z)
            return critic

On top of that, we define an Agent that is using the `AgentModel`.We consider an agent that can work both in stochastic or deterministic model, depending on the provided `agent_info`. In addition, the agent will produce an `agent_step` field to keep track of the computations.

.. code-block:: python

    class ReinforceAgent(Agent):
        def __init__(self,model=None, n_actions=None):
            super().__init__()
            self.model = model
            self.n_actions = n_actions


        def update(self,  state_dict):
            self.model.load_state_dict(state_dict)

        def __call__(self, state, observation,agent_info=None,history=None):
            """
            Executing one step of the agent
            """
            # Verify that the batch size is 1
            initial_state = observation["initial_state"]
            B = observation.n_elems()

            if agent_info is None:
                agent_info=DictTensor({"stochastic":torch.tensor([True]).repeat(B)})

            #We will store the agent step in the trajectories to illustrate how information can be propagated among multiple timesteps
            zero_step=DictTensor({"agent_step":torch.zeros(B).long()})
            if state is None:
                # if state is None, it means that the agent does not have any internal state. The internal state thus has to be initialized
                state = zero_step
            else:
                #We initialize the agent_step only for trajectory where an initial_state is met
                state = masked_dicttensor(state,zero_step,observation["initial_state"])
            #We compute one score per possible action
            action_proba = self.model(observation["frame"])

            #We sample an action following the distribution
            dist = torch.distributions.Categorical(action_proba)
            action_sampled = dist.sample()

            #Depending on the agent_info variable that tells us if we are in 'stochastic' or 'deterministic' mode, we keep the sampled action, or compute the action with the max score
            action_max = action_proba.max(1)[1]
            smask=agent_info["stochastic"].float()
            action=masked_tensor(action_max,action_sampled,agent_info["stochastic"])


            new_state = DictTensor({"agent_step": state["agent_step"] + 1})

            agent_do = DictTensor(
                {"action": action, "action_probabilities": action_proba}
            )

            return state, agent_do, new_state

Note that an `Agent` can produce any field in `agent_do` or `agent_state` but the produced field must be always the same, and of the same dimension.

Creating the learning Loop
--------------------------

To create the learning loop (see `reinforce.py`), the key element is the batcher which will sample episodes with multiple agents on multiple environments at the same time.
We use an `EpisodeBatcher` in our case to sample complete episodes. Such a batcher needs multiple parameters when created, and more particularly the functions and argument to create an `Agent` and a `rlstructures.VecEnv`.
These functions are usually declared in the main file (see `main_reinforce.py`) to avoid `pickle` problems in `spawn` multiprocessing mode.

We create the batcher as follows:

.. code-block:: python

    self.train_batcher=EpisodeBatcher(
            n_timesteps=self.config["max_episode_steps"],
            n_slots=self.config["n_envs"]*self.config["n_threads"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name":self.config["env_name"]
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_threads=self.config["n_threads"],
            seeds=[self.config["env_seed"]+k*10 for k in range(self.config["n_threads"])],
        )

The `n_timesteps` is the maximum size of the episode. `n_slots` is the maximum number of episodes that will be acquired simultaneously. In our case, we are using environments that each contain `n_envs` gym instances, and `n_threads` processes such that `n_envs * n_threads` episodes will be sampled at each iteration.
The `seeds` argument is used to choose the seed of the environment in each process, so we have as many seeds as `n_threads`

Now that we have a batcher, we can acquire `n_episodes = n_envs * n_threads` episodes through `batcher.execute`. Since `n_episodes` will be acquired simultaneously, we have to provide `n_episodes` agent information. In our case, we want all the agents to be in `stochastic` mode.

.. code-block:: python

    n_episodes=self.config["n_envs"]*self.config["n_threads"]
    agent_info=DictTensor({"stochastic":torch.tensor([True]).repeat(n_episodes)})
    self.train_batcher.execute(n_episodes=n_episodes,agent_info=agent_info)

Then episodes can be acquired as follows:

.. code-block:: python

    trajectories=self.train_batcher.get(blocking=True)

Here, the `get` function is in blocking mode, so the process will wait until the episodes have been acquired.

Computing the Reinforce Loss
------------------------

Now, we have trajectories on which we can compute a loss. The trajectories are a `TemporalDictTensor`, and each episode may be of different length (see `TemporalDictTensor.lengths` and `TemporalDictTensor.mask()`)

To compute the loss in REINFORCE, we first have to compute the cumulated discounted future reward. Note that the reward obtained by the action at time `t` is received in the observation at time `t+1`, and thus can be accessed throughg `trajectories["_reward"]` (don't forget that the prefix `_` corresponds to the state of the system at time `t+1`)

.. code-block:: python

            #First, we want to compute the cumulated reward per trajectory
            #The reward is a t+1 in each iteration (since it is btained after the aaction), so we use the '_reward' field in the trajectory
            # The 'reward' field corresopnds to the reward at time t
            reward=trajectories["_reward"]

            #We get the mask that tells which transition is in a trajectory (1) or not (0)
            mask=trajectories.mask()

            #We remove the reward values that are not in the trajectories
            reward=reward*mask

            #We compute the future cumulated reward at each timestep (by reverse computation)
            max_length=trajectories.lengths.max().item()
            cumulated_reward=torch.zeros_like(reward)
            cumulated_reward[:,max_length-1]=reward[:,max_length-1]
            for t in range(max_length-2,-1,-1):
                cumulated_reward[:,t]=reward[:,t]+self.config["discount_factor"]*cumulated_reward[:,t+1]

Now, we have to compute the action probabilities to be able to compute the gradient:

.. code-block:: python

            action_probabilities=[]
            for t in range(max_length):
                proba=self.learning_model(trajectories["frame"][:,t])
                action_probabilities.append(proba.unsqueeze(1)) # We append the probability, and introduces the temporal dimension (2nde dimension)
            action_probabilities=torch.cat(action_probabilities,dim=1) #Now, we have a B x T x n_actions tensor

And the same for the baseline:

.. code-block:: python

            baseline=[]
            for t in range(max_length):
                b=self.baseline_model(trajectories["frame"][:,t])
                baseline.append(b.unsqueeze(1))
            baseline=torch.cat(baseline,dim=1).squeeze(-1) #Now, we have a B x T tensor

At last, we can compute the baseline loss, the reinforce loss and the entropy loss easily (but don't forget to use the mask to consider only elements that are in each episodes since the episodes are of variable length)

.. code-block:: python

            #We compute the baseline loss
            baseline_loss=(baseline-cumulated_reward)**2
            #We sum the loss for each episode (considering the mask)
            baseline_loss= (baseline_loss*mask).sum(1)/mask.sum(1)
            #We average the loss over all the trajectories
            avg_baseline_loss = baseline_loss.mean()

            #We do the same on the reinforce loss
            action_distribution=torch.distributions.Categorical(action_probabilities)
            log_proba=action_distribution.log_prob(trajectories["action"])
            reinforce_loss = log_proba * (cumulated_reward-baseline).detach()
            reinforce_loss = (reinforce_loss*mask).sum(1)/mask.sum(1)
            avg_reinforce_loss=reinforce_loss.mean()

            #We compute the entropy loss
            entropy=action_distribution.entropy()
            entropy=(entropy*mask).sum(1)/mask.sum(1)
            avg_entropy=entropy.mean()

Remarks
-------

* Note that, once the model is updated, the parameters of the model have to be transmitted to the batcher since the batcher is running in another process.
* Note also that, easily, the loss computation can be made on GPU (running batcher on GPUs is more complicated)


Main function
-------------

Now, we can write the main function (see `main_reinforce.py`)
