#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import copy
import numpy as np

from rlstructures import VecEnv
from rlstructures import DictTensor
from rlstructures.env_wrappers.env_utils import format_frame
import gym.spaces

class GymEnv(VecEnv):
    """
    A wrapper for gym env
    """
    def __init__(self, gym_env=None, seed=None):
        super().__init__()
        assert not seed is None
        assert type(gym_env) is list
        self.gym_envs=gym_env
        for k in range(len(self.gym_envs)):
            self.gym_envs[k].seed(seed + k)
        self.action_space = self.gym_envs[0].action_space
        self.observation_space = self.gym_envs[0].observation_space

    def n_envs(self):
        return len(self.gym_envs)

    def reset(self,env_info=DictTensor({})):
        N = self.n_envs()
        self.envs_running = torch.arange(N)
        reward = torch.zeros(N)

        last_action=None
        if (isinstance(self.gym_envs[0].action_space,gym.spaces.Discrete)):
            last_action = torch.zeros(N, dtype=torch.int64)
        else:
            a=self.gym_envs[0].action_space.sample()
            a=torch.tensor(a).unsqueeze(0).repeat(N,1)
            last_action = a

        done = torch.zeros(N).bool()
        initial_state = torch.ones(N).bool()
        frames = None
        if (env_info.empty()):
            frames = [format_frame(e.reset()) for e in self.gym_envs]
        else:
            frames=[]
            for n in range(len(self.gym_envs)):
                v={k:env_info[k][n].tolist() for k in env_info.keys()}
                frames.append(format_frame(self.gym_envs[n].reset(env_info=v)))
        _frames = []
        for f in frames:
            if isinstance(f, torch.Tensor):
                _frames.append({"frame": f})
            else:
                _frames.append(f)
        frames = [DictTensor(_f) for _f in _frames]
        frames = DictTensor.cat(frames)
        frames.set("reward", reward)
        frames.set("done", done)
        frames.set("initial_state", initial_state)
        frames.set("last_action", last_action)
        return frames, self.envs_running

    def step(self, policy_output):
        assert policy_output.n_elems() == self.envs_running.size()[0]
        outputs = policy_output.unfold()
        alls = []
        env_run = {}
        for b in range(len(outputs)):
            idx_env = self.envs_running[b]
            action = policy_output["action"][b]
            last_action=action
            if (isinstance(self.gym_envs[0].action_space,gym.spaces.Discrete)):
                action=action.item()
                last_action=last_action.unsqueeze(0)
            else:
                action=action.tolist()
                last_action=last_action.unsqueeze(0)

            initial_state = torch.tensor([False])
            act = action

            frame, reward, done, unused_info = self.gym_envs[idx_env].step(act)
            reward = torch.tensor([reward])
            frame = format_frame(frame)
            if isinstance(frame, torch.Tensor):
                frame = {"frame": frame}
            if not done:
                env_run[b] = idx_env

            done = torch.tensor([done])
            r = DictTensor(
                {
                    "reward": reward,
                    "done": done,
                    "initial_state": initial_state,
                    "last_action": last_action,
                    **frame,
                }
            )
            alls.append(r)

        d = DictTensor.cat(alls)

        keys = []
        values = []
        for key, value in env_run.items():
            keys.append(key)
            values.append(value)
        dd = d.index(torch.tensor(keys).long())
        old_envs_running = self.envs_running
        self.envs_running = torch.tensor(values)
        return (d, old_envs_running), (dd, self.envs_running)

    def close(self):
        [g.close() for g in self.gym_envs]

    def specs_input(self):
        return {
            "action": {
                "size": torch.tensor([]).size(),
                "dtype": torch.zeros(1).long().dtype,
            }
        }


class GymEnvInf(VecEnv):
    """
    A wrapper for gym env that automaitcally reset each stopping instance
    """
    def __init__(self, gym_env=None, seed=None):
        super().__init__()
        assert not seed is None
        assert type(gym_env) is list
        self.gym_envs=gym_env
        for k in range(len(self.gym_envs)):
            self.gym_envs[k].seed(seed + k)
        self.action_space = self.gym_envs[0].action_space
        self.observation_space = self.gym_envs[0].observation_space

    def n_envs(self):
        return len(self.gym_envs)

    def reset(self,env_info=DictTensor({})):
        N = self.n_envs()
        reward = torch.zeros(N)

        last_action=None
        if (isinstance(self.gym_envs[0].action_space,gym.spaces.Discrete)):
            last_action = torch.zeros(N, dtype=torch.int64)
        else:
            a=self.gym_envs[0].action_space.sample()
            a=torch.tensor(a).unsqueeze(0).repeat(N,1)
            last_action = a


        done = torch.zeros(N).bool()
        initial_state = torch.ones(N).bool()
        self.env_info=env_info
        frames = None
        if (env_info.empty()):
            frames = [format_frame(e.reset()) for e in self.gym_envs]
        else:
            frames=[]
            for n in range(len(self.gym_envs)):
                v={k:env_info[k][n].tolist() for k in env_info.keys()}
                frames.append(format_frame(self.gym_envs[n].reset(env_info=v)))
        _frames = []
        for f in frames:
            if isinstance(f, torch.Tensor):
                _frames.append({"frame": f})
            else:
                _frames.append(f)
        frames = [DictTensor(_f) for _f in _frames]
        frames = DictTensor.cat(frames)
        frames.set("reward", reward)
        frames.set("done", done)
        frames.set("initial_state", initial_state)
        frames.set("last_action", last_action)
        return frames, torch.arange(N)

    def step(self, policy_output):
        assert policy_output.n_elems() == self.n_envs()
        outputs = policy_output.unfold()
        alls = []
        alls_after = []
        env_run = {}
        for b in range(len(outputs)):
            action = policy_output["action"][b]
            last_action = action
            if (isinstance(self.gym_envs[0].action_space,gym.spaces.Discrete)):
                action=action.item()
                last_action=last_action.unsqueeze(0)
            else:
                action=action.tolist()
                last_action=last_action.unsqueeze(0)

            initial_state = torch.tensor([False])
            act = action

            frame, reward, done, unused_info = self.gym_envs[b].step(act)
            reward = torch.tensor([reward])
            frame = format_frame(frame)
            if isinstance(frame, torch.Tensor):
                frame = {"frame": frame}

            done = torch.tensor([done])
            r = DictTensor(
                {
                    "reward": reward,
                    "done": done,
                    "initial_state": initial_state,
                    "last_action": last_action,
                    **frame,
                }
            )
            alls.append(r)

            if done:
                if "set" in dir(self.gym_envs[b]):
                    self.gym_envs[b].set(self.env_info[b])


                if self.env_info.empty():
                    frame = self.gym_envs[b].reset()
                else:
                    v={k:env_info[k][b].tolist() for k in env_info.keys()}
                    frame = self.gym_envs[b].reset(env_info=v)

                frame = format_frame(frame)
                if isinstance(frame, torch.Tensor):
                    frame = {"frame": frame}

                last_action=None
                if (isinstance(self.gym_envs[0].action_space,gym.spaces.Discrete)):
                    last_action = torch.zeros(1, dtype=torch.int64)
                else:
                    a=self.gym_envs[0].action_space.sample()
                    a=torch.tensor([a])
                    last_action = a

                initial_state = torch.tensor([True])
                reward = torch.tensor([0.0])
                r = DictTensor(
                    {
                        "reward": reward,
                        "done": done,
                        "initial_state": initial_state,
                        "last_action": last_action,
                        **frame,
                    }
                )
                alls_after.append(r)
            else:
                alls_after.append(r)

        next_observation = DictTensor.cat(alls)
        next_observation_next_slot = DictTensor.cat(alls_after)
        return (
            (next_observation, torch.arange(self.n_envs())),
            (next_observation_next_slot, torch.arange(self.n_envs())),
        )

    def close(self):
        [g.close() for g in self.gym_envs]

    def specs_input(self):
        return {
            "action": {
                "size": torch.tensor([]).size(),
                "dtype": torch.zeros(1).long().dtype,
            }
        }
