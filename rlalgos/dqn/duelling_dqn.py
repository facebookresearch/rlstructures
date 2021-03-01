#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlalgos.logger import Logger, TFLogger
from rlstructures import DictTensor, TemporalDictTensor, RL_Batcher
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
from rlstructures.batchers import Batcher, EpisodeBatcher
import math

# class ReplayBuffer:
#     def __init__(self,N):
#         self.N=N
#         self.buffer=None

#     def _init_buffer(self,trajectories):
#         self.buffer={}
#         for k in trajectories.keys():
#             dtype=trajectories[k].dtype
#             size=trajectories[k].size()
#             b_size=(self.N,)+size[2:]
#             self.buffer[k]=torch.zeros(*b_size,dtype=dtype)
#         self.pos=0
#         self.full=False

#     def write(self,trajectories):
#         rs={}
#         new_pos=None
#         for k in trajectories.keys():
#             v=trajectories[k]
#             size=v.size()
#             b_size=(size[0]*size[1],)+size[2:]
#             v=v.reshape(*b_size)
#             n=v.size()[0]
#             overhead=self.N-(self.pos+n)
#             if new_pos is None:
#                 new_pos=torch.arange(n)+self.pos
#                 mask=new_pos.ge(self.N).float()
#                 nidx=torch.arange(n)+self.pos-self.N
#                 new_pos=(new_pos*(1-mask)+mask*(nidx)).long()

#             self.buffer[k][new_pos]=v
#         self.pos=self.pos+n
#         if self.pos>=self.N:
#             self.pos=self.pos-self.N
#             self.full=True
#         assert self.pos<self.N

#     def size(self):
#         if self.full:
#             return self.N
#         else:
#             return self.pos


#     def push(self,trajectories):
#         '''
#         Add transitions to the replay buffer
#         '''
#         max_length=trajectories.lengths.max().item()
#         assert trajectories.lengths.eq(max_length).all()
#         if self.buffer is None:
#             self._init_buffer(trajectories)
#         self.write(trajectories)

#     def sample(self,n=1):
#         limit=self.pos
#         if self.full:
#             limit=self.N
#         transitions=torch.randint(0,high=limit,size=(n,))
#         d={k:self.buffer[k][transitions] for k in self.buffer}
#         return DictTensor(d)


"""
A DQN Implementation with prioritized experience replay. It implements also double an duelling Q-Learning

Parameters:
    "environment/env_name": openai gym environment name
    "n_envs": Nb single envs per process
    "max_episode_steps": Max Number of environment steps (for evaluation)
    "discount_factor": The discount factor
    "epsilon_greedy_max": The epsilon (greedy) value at the beginning of training
    "epsilon_greedy_min": The epsilon (greedy) value at the end of the linear decay
    "epsilon_min_epoch": The epoch at which the min epsilon greedy value is reached
    "replay_buffer_size": The nubmer of transitions in the replay buffer
    "n_batches": The size of the batch when updating Q
    "initial_buffer_epochs": The number of epochs made to initialize the replay buffer with a uniform policiy
    "qvalue_epochs": The number of q updates between two acquistion of transitions
    "batch_timesteps": The number of timesteps executed at each acquisition step
    "use_duelling": False,
    "use_double":False,
    "lr": 0.00001,
    "n_processes": Number of learning processes for acquisition
    "n_evaluation_processes": Number of processes for evaluation
    "verbose": False,
    "n_evaluation_envs": Number of single envs per evaluation process
    "time_limit": in seconds
    "env_seed": 42,
    "clip_grad": 0.0,
    "learner_device": "cpu",
    "as_fast_as_possible": if True: execute as many update iteration as possible between two acquisitions. The min numboer is q_epochs

    "update_target_hard":True,
    "update_target_epoch":1000,
    "update_target_tau": 0.005,

    "buffer/alpha":0.6,
    "buffer/beta":0.4,

    "logdir":"./results",
    "save_every":100,
"""


class ReplayBuffer:
    def __init__(self, N):
        self.N = N
        self.buffer = None
        self.priorities = None

    def _init_buffer(self, trajectories):
        print("\tCreating relpay buffer of size ", self.N)
        self.buffer = {}
        for k in trajectories.keys():
            dtype = trajectories[k].dtype
            size = trajectories[k].size()
            b_size = (self.N,) + size[2:]
            self.buffer[k] = torch.zeros(*b_size, dtype=dtype)
        self.priorities = torch.zeros(self.N)
        self.pos = 0
        self.full = False

    def write(self, trajectories):
        limit = self.pos
        if self.full:
            limit = self.N

        rs = {}
        new_pos = None
        for k in trajectories.keys():
            v = trajectories[k]
            size = v.size()
            b_size = (size[0] * size[1],) + size[2:]
            v = v.reshape(*b_size)
            n = v.size()[0]
            overhead = self.N - (self.pos + n)
            if new_pos is None:
                new_pos = torch.arange(n) + self.pos
                mask = new_pos.ge(self.N).float()
                nidx = torch.arange(n) + self.pos - self.N
                new_pos = (new_pos * (1 - mask) + mask * (nidx)).long()

            self.buffer[k][new_pos] = v
        if limit == 0:
            self.priorities[new_pos] = 1.0
        else:
            self.priorities[new_pos] = self.priorities[:limit].max()

        self.pos = self.pos + n
        if self.pos >= self.N:
            self.pos = self.pos - self.N
            self.full = True
        assert self.pos < self.N

    def size(self):
        if self.full:
            return self.N
        else:
            return self.pos

    def push(self, trajectories):
        """
        Add transitions to the replay buffer
        """
        max_length = trajectories.lengths.max().item()
        assert trajectories.lengths.eq(max_length).all()
        if self.buffer is None:
            self._init_buffer(trajectories)
        self.write(trajectories)

    def sample(self, n=1, alpha=0.0, beta=0.0):
        limit = self.pos
        if self.full:
            limit = self.N
        if alpha == 0 and beta == 0:
            w = 1.0 / limit
            idxs = torch.randint(limit, size=(n,))
            d = {k: self.buffer[k][idxs] for k in self.buffer}
            return DictTensor(d), None, None

        distribution = self.priorities[:limit] ** alpha
        distribution = distribution / distribution.sum()
        # print(distribution)
        d = torch.distributions.Categorical(distribution.unsqueeze(0).repeat(n, 1))
        transitions = d.sample()
        del d
        imax = self.priorities.max(0)[1]
        weights = (limit * distribution[transitions]) ** (-beta)
        weights /= weights.max()

        d = {k: self.buffer[k][transitions] for k in self.buffer}
        # print(weights)
        return DictTensor(d), transitions, weights

    def update_priorities(self, transitions, priorities):
        self.priorities[transitions] = priorities


class DQN:
    def __init__(self, config, create_env, create_agent):
        self.config = config

        # Creation of the Logger (that saves in tensorboard and CSV)
        self.logger = TFLogger(
            log_dir=self.config["logdir"],
            hps=self.config,
            save_every=self.config["save_every"],
        )

        self._create_env = create_env
        self._create_agent = create_agent

    def _state_dict(self, model, device):
        sd = model.state_dict()
        for k, v in sd.items():
            sd[k] = v.to(device)
        return sd

    def run(self):
        env = self._create_env(
            self.config["n_envs"],
            seed=0,
            **{k: self.config[k] for k in self.config if k.startswith("environment/")}
        )
        self.n_actions = env.action_space.n
        self.obs_shape = env.reset()[0]["frame"].size()
        del env

        # Create the agent model
        self.learning_model = self._create_model()
        self.target_model = copy.deepcopy(self.learning_model)

        # Create one agent for loss computation (see get_loss)
        self.agent = self._create_agent(
            n_actions=self.n_actions, model=self.learning_model
        )

        model = copy.deepcopy(self.learning_model)
        self.train_batcher = RL_Batcher(
            n_timesteps=self.config["batch_timesteps"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "mode": "train",
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                **{
                    k: self.config[k]
                    for k in self.config
                    if k.startswith("environment/")
                },
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_processes=self.config["n_processes"],
            seeds=[
                self.config["env_seed"] + k * 10
                for k in range(self.config["n_processes"])
            ],
            agent_info=DictTensor({"epsilon": torch.zeros(1)}),
            env_info=DictTensor({}),
        )

        model = copy.deepcopy(self.learning_model)
        self.evaluation_batcher = RL_Batcher(
            n_timesteps=self.config["max_episode_steps"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "mode": "evaluation",
                "max_episode_steps": self.config["max_episode_steps"],
                "n_envs": self.config["n_evaluation_envs"],
                **{
                    k: self.config[k]
                    for k in self.config
                    if k.startswith("environment/")
                },
            },
            agent_args={"n_actions": self.n_actions, "model": model},
            n_processes=self.config["n_evaluation_processes"],
            seeds=[
                self.config["env_seed"] * 10 + k * 10
                for k in range(self.config["n_evaluation_processes"])
            ],
            agent_info=DictTensor({"epsilon": torch.zeros(1)}),
            env_info=DictTensor({}),
        )

        self.replay_buffer = ReplayBuffer(self.config["replay_buffer_size"])
        device = torch.device(self.config["learner_device"])
        self.learning_model.to(device)
        self.target_model.to(device)
        optimizer = getattr(torch.optim, self.config["optim"])(
            self.learning_model.parameters(), lr=self.config["lr"]
        )

        self.evaluation_batcher.update(
            self._state_dict(self.learning_model, torch.device("cpu"))
        )

        n_episodes = self.config["n_envs"] * self.config["n_processes"]
        agent_info = DictTensor({"epsilon": torch.ones(n_episodes).float()})
        self.train_batcher.reset(agent_info=agent_info)

        logging.info("Sampling initial transitions")
        for k in range(self.config["initial_buffer_epochs"]):
            self.train_batcher.execute()
            trajectories, n = self.train_batcher.get(blocking=True)
            assert not n == 0
            self.replay_buffer.push(trajectories.trajectories)

        self.iteration = 0

        n_episodes = (
            self.config["n_evaluation_envs"] * self.config["n_evaluation_processes"]
        )
        self.evaluation_batcher.reset(
            agent_info=DictTensor({"epsilon": torch.zeros(n_episodes).float()})
        )
        # self.evaluation_batcher.reset(agent_info=DictTensor({"epsilon":torch.zeros(n_episodes)}))
        self.evaluation_batcher.execute()

        logging.info("Starting Learning")
        _start_time = time.time()

        produced = 0
        consumed = 0
        n_interactions = self.replay_buffer.size()
        self.target_model.load_state_dict(self.learning_model.state_dict())
        cumulated_reward = torch.zeros(
            self.config["n_envs"] * self.config["n_processes"]
        )

        epsilon_step = (
            self.config["epsilon_greedy_max"] - self.config["epsilon_greedy_min"]
        ) / self.config["epsilon_min_epoch"]
        self.epsilon = self.config["epsilon_greedy_max"] - epsilon_step * self.iteration
        self.epsilon = max(self.epsilon, self.config["epsilon_greedy_min"])
        self.logger.add_scalar("epsilon", self.epsilon, self.iteration)
        n_episodes = self.config["n_envs"] * self.config["n_processes"]
        self.train_batcher.update(
            self._state_dict(self.learning_model, torch.device("cpu"))
        )
        self.train_batcher.execute(
            agent_info=DictTensor(
                {"epsilon": torch.tensor([self.epsilon]).repeat(n_episodes).float()}
            )
        )
        print("Go learning...")
        while time.time() - _start_time < self.config["time_limit"]:
            trajectories, n = self.train_batcher.get(
                blocking=not self.config["as_fast_as_possible"]
            )

            if not trajectories is None:
                epsilon_step = (
                    self.config["epsilon_greedy_max"]
                    - self.config["epsilon_greedy_min"]
                ) / self.config["epsilon_min_epoch"]
                self.epsilon = (
                    self.config["epsilon_greedy_max"] - epsilon_step * self.iteration
                )
                self.epsilon = max(self.epsilon, self.config["epsilon_greedy_min"])

                self.logger.add_scalar("epsilon", self.epsilon, self.iteration)
                n_episodes = self.config["n_envs"] * self.config["n_processes"]
                self.train_batcher.update(
                    self._state_dict(self.learning_model, torch.device("cpu"))
                )
                self.train_batcher.execute(
                    agent_info=DictTensor(
                        {
                            "epsilon": torch.tensor([self.epsilon])
                            .repeat(n_episodes)
                            .float()
                        }
                    )
                )

                reward = trajectories.trajectories["_observation/reward"]
                _is = trajectories.trajectories["observation/initial_state"]
                crs = []
                for t in range(reward.size(1)):
                    cr = cumulated_reward[_is[:, t]]
                    for ii in range(cr.size()[0]):
                        # print("CR = ",cr[ii].item())
                        crs.append(cr[ii].item())
                    cumulated_reward = (
                        torch.zeros_like(cumulated_reward) * _is[:, t].float()
                        + (1 - _is[:, t].float()) * cumulated_reward
                    )
                    cumulated_reward += reward[:, t]
                if len(crs) > 0:
                    self.logger.add_scalar(
                        "train_cumulated_reward", np.mean(crs), self.iteration
                    )

                assert n == self.config["n_envs"] * self.config["n_processes"]
                self.replay_buffer.push(trajectories.trajectories)
                produced += trajectories.trajectories.lengths.sum().item()
                self.logger.add_scalar(
                    "stats/replay_buffer_size",
                    self.replay_buffer.size(),
                    self.iteration,
                )

            # avg_reward = 0
            assert self.config["qvalue_epochs"] > 0
            for k in range(self.config["qvalue_epochs"]):
                optimizer.zero_grad()
                alpha = self.config["buffer/alpha"]
                beta = self.config["buffer/beta"]
                transitions, idx, weights = self.replay_buffer.sample(
                    n=self.config["n_batches"], alpha=alpha, beta=beta
                )
                consumed += transitions.n_elems()
                dt = self.get_loss(transitions, device)
                _loss = None

                if alpha == 0 and beta == 0:
                    _loss = dt["q_loss"].to(self.config["learner_device"]).mean()
                else:
                    self.replay_buffer.update_priorities(
                        idx, dt["q_loss"].sqrt().detach().to("cpu")
                    )
                    _loss = (
                        dt["q_loss"] * weights.to(self.config["learner_device"])
                    ).mean()

                self.logger.add_scalar("q_loss", _loss.item(), self.iteration)

                _loss.backward()
                if self.config["clip_grad"] > 0:
                    n = torch.nn.utils.clip_grad_norm_(
                        self.learning_model.parameters(), self.config["clip_grad"]
                    )
                    self.logger.add_scalar("grad_norm", n.item(), self.iteration)
                self.iteration += 1
                optimizer.step()

                if self.config["update_target_hard"]:
                    if self.iteration % self.config["update_target_epoch"] == 0:
                        self.target_model.load_state_dict(
                            self.learning_model.state_dict()
                        )
                else:
                    tau = self.config["update_target_tau"]
                    self.soft_update_params(self.learning_model, self.target_model, tau)

                if time.time() - _start_time > 600 and self.iteration % 1000 == 0:
                    self.logger.update_csv()

            tt = time.time()
            c_ps = consumed / (tt - _start_time)
            p_ps = produced / (tt - _start_time)
            # print(p_ps,c_ps)
            self.logger.add_scalar("speed/consumed_per_seconds", c_ps, self.iteration)
            self.logger.add_scalar(
                "speed/n_interactions", n_interactions + produced, self.iteration
            )
            self.logger.add_scalar("speed/produced_per_seconds", p_ps, self.iteration)
            self.evaluate()
        self.logger.update_csv()  # To save as a CSV file in logdir

        trajectories, n = self.train_batcher.get()

        self.train_batcher.close()
        self.evaluation_batcher.get()  # To wait for the last trajectories
        self.evaluation_batcher.close()
        self.logger.close()

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def evaluate(self, relaunch=True):

        evaluation_trajectories, n = self.evaluation_batcher.get(blocking=False)

        if evaluation_trajectories is None:
            return
        # print(evaluation_trajectories.trajectories.lengths)
        # assert n==0
        avg_reward = (
            (
                evaluation_trajectories.trajectories["_observation/reward"]
                * evaluation_trajectories.trajectories.mask()
            )
            .sum(1)
            .mean()
            .item()
        )
        self.logger.add_scalar("avg_reward", avg_reward, self.iteration)
        if self.config["verbose"]:
            print(
                "Iteration "
                + str(self.iteration)
                + ", Reward =  "
                + str(avg_reward)
                + ", Buffer size = "
                + str(self.replay_buffer.size())
            )

        if relaunch:
            self.evaluation_batcher.update(
                self._state_dict(self.learning_model, torch.device("cpu"))
            )
            n_episodes = (
                self.config["n_evaluation_envs"] * self.config["n_evaluation_processes"]
            )
            # self.evaluation_batcher.reset(agent_info=DictTensor({"epsilon":torch.zeros(n_episodes)}))
            self.evaluation_batcher.reset(
                agent_info=DictTensor({"epsilon": torch.zeros(n_episodes).float()})
            )
            self.evaluation_batcher.execute()
        return avg_reward

    def get_loss(self, transitions, device):
        transitions = transitions.to(device)
        B = transitions.n_elems()
        Bv = torch.arange(B).to(device)
        action = transitions["action/action"]
        reward = transitions["_observation/reward"]
        frame = transitions["observation/frame"]
        _frame = transitions["_observation/frame"]
        _done = transitions["_observation/done"].float()

        q = self.learning_model(frame)
        qa = q[Bv, action]

        # qp = self.learning_model(_frame).detach()
        _q_target = self.target_model(_frame).detach()
        _q_target_a = None
        if not self.config["use_double"]:
            actionp = _q_target.max(1)[1]
            _q_target_a = _q_target[Bv, actionp]
        else:
            qp = self.learning_model(_frame).detach()
            actionp = qp.max(1)[1]
            _q_target_a = _q_target[Bv, actionp]
        _target_value = (
            _q_target_a * (1 - _done) * self.config["discount_factor"] + reward
        )

        td = (_target_value - qa) ** 2
        dt = DictTensor(
            {
                "q_loss": td,
            }
        )
        return dt
