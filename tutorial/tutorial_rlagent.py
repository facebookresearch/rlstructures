#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from rlstructures import RL_Agent, DictTensor
import torch


class UniformAgent(RL_Agent):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def initial_state(self, agent_info, B):
        return DictTensor({"timestep": torch.zeros(B).long()})

    def __call__(self, state, observation, agent_info=None, history=None):
        B = observation.n_elems()

        scores = torch.randn(B, self.n_actions)
        probabilities = torch.softmax(scores, dim=1)
        actions = torch.distributions.Categorical(probabilities).sample()
        new_state = DictTensor({"timestep": state["timestep"] + 1})
        return DictTensor({"action": actions}), new_state


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


def create_env(max_episode_steps=100, seed=None):
    envs = []
    for k in range(4):
        e = gym.make("CartPole-v0")
        e.seed(seed)
        e = TimeLimit(e, max_episode_steps=max_episode_steps)
        envs.append(e)
    return GymEnv(envs, seed=10)


def create_agent(n_actions):
    return UniformAgent(n_actions)


if __name__ == "__main__":
    # We use spawn mode such that most of the environment will run in multiple processes
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    from rlstructures import RL_Batcher

    batcher = RL_Batcher(
        n_timesteps=100,
        create_agent=create_agent,
        create_env=create_env,
        agent_args={"n_actions": 2},
        env_args={"max_episode_steps": 100},
        n_processes=1,
        seeds=[42],
        agent_info=DictTensor({}),
        env_info=DictTensor({}),
    )

    batcher.reset()
    batcher.execute()
    trajectories, n_still_running_envs = batcher.get()

    print("Informations: ")
    print(trajectories, trajectories.info)
    print("Lengths of trajectories: ")
    print(trajectories.trajectories.lengths)
