#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from rlstructures.env_wrappers import GymEnv, GymEnvInf
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
import gym
from gym.wrappers import TimeLimit
from rlstructures import (
    RL_Batcher,
    RL_Agent,
    DictTensor,
    Trajectories,
    TemporalDictTensor,
    masked_dicttensor,
)

# Let us create an environment that contains multiple CartPole environments
def create_env(n_envs, max_episode_steps=None, seed=None):
    envs = []
    for k in range(n_envs):
        e = gym.make("CartPole-v0")
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnv(envs, seed)


# Let us create an environment that contains multiple CartPole environments
def create_autoreset_env(n_envs, max_episode_steps=None, seed=None):
    envs = []
    for k in range(n_envs):
        e = gym.make("CartPole-v0")
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnvInf(envs, seed)


# Let us create an agent that:
# * Choose the action specified in the agent_info["action"] argument, and use the episode timestep as an internalstate
class MyAgent(RL_Agent):
    def initial_state(self, agent_info, B):
        return DictTensor({"timestep": torch.zeros(B).long()})

    def __call__(self, state, observation, agent_info=None, history=None):
        B = observation.n_elems()
        # This line is used to reinitialize the internal state if the agent reach a new initial state
        state = masked_dicttensor(
            state, self.initial_state(agent_info, B), observation["initial_state"]
        )
        new_state = DictTensor({"timestep": state["timestep"] + 1})
        action = agent_info["user_action"]
        return DictTensor({"action": action}), new_state


def create_agent():
    return MyAgent()


# Now, let us play ot understand the data structures exacnahged by the different components
if __name__ == "__main__":
    # We use spawn mode such that most of the environment will run in multiple processes
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    # This batcher is used to sample complete episodes (it will stop after 100 timsteps, and the envs are limited to 100 timesteps)
    # Since the Agent uses a agent_info["user_action"] field, we need to tell it to the batcher (but can use a size of 1)
    n_envs = 4
    max_timesteps = 100
    n_processes = 2
    episode_batcher = RL_Batcher(
        n_timesteps=max_timesteps,
        create_agent=create_agent,
        create_env=create_env,
        agent_args={},
        env_args={"n_envs": n_envs, "max_episode_steps": max_timesteps},
        n_processes=n_processes,
        seeds=[42 + k for k in range(n_processes)],
        agent_info=DictTensor({"user_action": torch.zeros(1).long()}),
        env_info=DictTensor({}),
    )

    # Let gets episodes. The batcher has n_processes, each processes contains a GymEnv with n_envs environment ninstances.
    # The number of sampled trajectories will be n_envs*n_processes
    n_trajectories = n_envs * n_processes

    # We must transmit information to the agent (agent_info). One information has to be transmitted to each of the n_trajectories policies
    agent_info = DictTensor({"user_action": torch.zeros(n_trajectories).long()})

    # Now, reset, and execute
    episode_batcher.reset(agent_info=agent_info)
    episode_batcher.execute()
    trajectories, n = episode_batcher.get(blocking=True)

    # Since the batcher tried to acqire max_timesteps over environments that are limited to max_timesteps, the environmnets must have all ended during acquisition
    assert n == 0

    # The info field contain the agent_info, env_info, and initial state when calling the batcher
    print("Agent_info ", trajectories.info.truncate_key("agent_info/"))
    print("Agent_state ", trajectories.info.truncate_key("agent_state/"))

    # Now, the trajectories must have different length, since the different CartPole instances may have stopped at different timesteps
    lengths = trajectories.trajectories.lengths
    print("Lengths ", lengths)
    # and the mask provide the same information
    print("Mask ", trajectories.trajectories.mask())

    B = trajectories.trajectories.n_elems()

    # The first state of the trajectories are surely initial_state
    print(
        "Initial state = ", trajectories.trajectories["observation/initial_state"][:, 0]
    )

    # The t+1 state of the last timestep are surely final states
    print(
        "Done =",
        trajectories.trajectories["_observation/done"][torch.arange(B), lengths - 1],
    )

    # Actions are all 0 (this is what we decided when building agent_info)
    print("Actions =", trajectories.trajectories["action/action"])

    # We can provide one information different per agent:
    agent_info = DictTensor({"user_action": torch.randn(n_trajectories).gt(0.0).long()})
    episode_batcher.reset(agent_info=agent_info)
    episode_batcher.execute()
    trajectories, n = episode_batcher.get(blocking=True)
    print("Actions =", trajectories.trajectories["action/action"])
    episode_batcher.close()
    del episode_batcher
    # ============================================ Splitting trajectories
    # Now, we can acquire episodes "piece by piece", each piece being of size n:
    n = 4
    slot_batcher = RL_Batcher(
        n_timesteps=n,
        create_agent=create_agent,
        create_env=create_env,
        agent_args={},
        env_args={"n_envs": n_envs, "max_episode_steps": max_timesteps},
        n_processes=n_processes,
        seeds=[42 + k for k in range(n_processes)],
        agent_info=DictTensor({"user_action": torch.zeros(1).long()}),
        env_info=DictTensor({}),
    )
    agent_info = DictTensor({"user_action": torch.zeros(n_trajectories).long()})
    slot_batcher.reset(agent_info=agent_info)
    slot_batcher.execute()
    trajectories, n = slot_batcher.get(blocking=True)
    # Some environment my have stopped, but not all
    print("Number of still running environments : ", n)

    # Since all episodes did not stop, I can acquire the remaining steps over running environments
    slot_batcher.execute()
    trajectories, n = slot_batcher.get(blocking=True)
    print(
        "The initial state of the running agent in this slot is ",
        trajectories.info.truncate_key("agent_state/"),
    )
    print("Number of still running environments : ", n)

    slot_batcher.execute()
    trajectories, n = slot_batcher.get(blocking=True)
    print(
        "The trajectories contain the execution of ",
        trajectories.info.n_elems(),
        " agents",
    )
    print(
        "The initial state of the running agent in this slot is ",
        trajectories.info.truncate_key("agent_state/"),
    )
    print("Number of still running environments : ", n)
    slot_batcher.close()
    del slot_batcher

    # =========================== Autoreset envs
    # Now, I will create a batcher with autoreset envs
    n = 4
    slot_batcher = RL_Batcher(
        n_timesteps=n,
        create_agent=create_agent,
        create_env=create_autoreset_env,
        agent_args={},
        env_args={"n_envs": n_envs, "max_episode_steps": max_timesteps},
        n_processes=n_processes,
        seeds=[42 + k for k in range(n_processes)],
        agent_info=DictTensor({"user_action": torch.zeros(1).long()}),
        env_info=DictTensor({}),
    )

    agent_info = DictTensor({"user_action": torch.zeros(n_trajectories).long()})
    slot_batcher.reset(agent_info=agent_info)

    # The environments never stop !
    for k in range(10):
        slot_batcher.execute()
        trajectories, n = slot_batcher.get(blocking=True)
        print("Number of still running environments : ", n)
        print("Lengts ", trajectories.trajectories.lengths)

    # Some transitions have their t+1 state as a final_state of one episode
    # In that case, the observation in the next transition is an initial_state
    print("Done ", trajectories.trajectories["_observation/done"])
    print("Initial state ", trajectories.trajectories["observation/initial_state"])
