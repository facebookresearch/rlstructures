#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.multiprocessing as mp
import rlstructures.logging as logging
from rlstructures import TemporalDictTensor,DictTensor


class Buffer:
    def get_free_slots(self, k):
        raise NotImplementedError

    def set_free_slots(self, s):
        raise NotImplementedError

    def write(self, slots, variables):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def get_trajectories(self, trajectories, erase=True):
        raise NotImplementedError


class LocalBuffer(Buffer):
    """
    Defines a shared buffer to store trajectories / transitions
    The buffer is structured as nslots of size s_slots for each possible variable
    """

    def __init__(
        self,
        n_slots=None,
        s_slots=None,
        specs_agent_state=None,
        specs_agent_output=None,
        specs_environment=None,
        device=torch.device("cpu"),
    ):
        """
        Init a new buffer

        Args:
            n_slots (int): the number of slots
            s_slots (int): the size of each slot (temporal dimension)
            specs (dict): The description of the variable to store in the buffer
        """
        self._device = device
        self.buffers = {}
        self.n_slots = n_slots
        self.s_slots = s_slots

        # Creation of the storage buffers
        nspecs_agent_state = {"_" + k: specs_agent_state[k] for k in specs_agent_state}
        nspecs_env = {"_" + k: specs_environment[k] for k in specs_environment}
        specs = {
            **specs_agent_state,
            **specs_agent_output,
            **specs_environment,
            **nspecs_agent_state,
            **nspecs_env,
            "position_in_slot":{"size": torch.Size([]), "dtype": torch.int64}
        }

        for n in specs:
            size = (n_slots, s_slots) + specs[n]["size"]
            logging.getLogger("buffer").debug(
                "Creating buffer for '"
                + n
                + "' of size "
                + str(size)
                + " and type "
                + str(specs[n]["dtype"])
            )
            assert not n in self.buffers, "Same key is used by the agent and the env"
            self.buffers[n] = (
                torch.zeros(size, dtype=specs[n]["dtype"])
                .to(self._device)
                .share_memory_()
            )
        self.position_in_slot = (
            torch.zeros(n_slots).to(self._device).long().share_memory_()
        )
        self._free_slots_queue = mp.Queue()
        self._free_slots_queue.cancel_join_thread()
        for i in range(n_slots):
            self._free_slots_queue.put(i, block=True)
        self._full_slots_queue = mp.Queue()
        self._full_slots_queue.cancel_join_thread()

    def device(self):
        return self._device

    def get_free_slots(self, k):
        """
        Returns k available slots. Wait until enough slots are free
        """
        assert k > 0
        x = [self._free_slots_queue.get() for i in range(k)]
        for i in x:
            self.position_in_slot[i] = 0
        # logging.getLogger("buffer").debug("GET FREE " + str(x))
        return x

    def set_free_slots(self, s):
        """
        Tells the buffer that it can reuse the given slots 
        :param s may be one slot (int) or multiple slots (list of int)
        """
        assert not s is None
        if isinstance(s, int):
            self._free_slots_queue.put(s)
        else:
            for ss in s:
                self._free_slots_queue.put(ss)
        # logging.getLogger("buffer").debug("SET FREE " + str(s))

    def write(self, slots, variables):
        if not variables.device() == self._device:
            variables = variables.to(self._device)

        slots = torch.tensor(slots).to(self._device)
        assert variables.n_elems() == len(slots)
        positions = self.position_in_slot[slots]
        a = torch.arange(len(slots)).to(self._device)
        # print("Write in "+str(slot)+" at positions "+str(position))
        for n in variables.keys():
            # assert variables[n].size()[0] == 1
            # print(self.buffers[n][slots].size())
            self.buffers[n][slots, positions] = variables[n][a].detach()
        self.position_in_slot[slots] += 1

    def is_slot_full(self, slot):
        """ 
        Returns True of a slot is full
        """
        return self.position_in_slot[slot] == self.s_slots

    def get_single(self,slots,position):
        assert isinstance(slots, list)
        assert isinstance(slots[0], int)
        idx = torch.tensor(slots).to(self._device).long()
        d={k:self.buffers[k][idx,position] for k in self.buffers}
        return DictTensor(d)

    def close(self):
        """
        Close the buffer
        """
        self._free_slots_queue.close()
        self._full_slots_queue.close()

    def get_single_slots(self, slots, erase=True):
        assert isinstance(slots, list)
        assert isinstance(slots[0], int)
        idx = torch.tensor(slots).to(self._device).long()
        lengths = self.position_in_slot[idx]
        ml = lengths.max().item()
        v = {k: self.buffers[k][idx, :ml] for k in self.buffers}
        if erase:
            self.set_free_slots(slots)
        return TemporalDictTensor(v, lengths)

    def get_multiple_slots(self, trajectories, erase=True):
        """
        Return the concatenation of multiple slots. This function is not well optimized and could be fasten
        """
        assert isinstance(trajectories, list) or isinstance(trajectories, tuple)
        assert isinstance(trajectories[0], list)
        assert isinstance(trajectories[0][0], int)
        # 1: Unify the size of all trajectories....
        max_l = 0
        for traj in trajectories:
            max_l = max(max_l, len(traj))
        ntrajectories = []
        for traj in trajectories:
            while not len(traj) == max_l:
                traj.append(None)
            ntrajectories.append(traj)

        # 2: Copy the content
        length = torch.zeros(len(ntrajectories)).to(self._device).long()
        tensors = []
        for k in range(max_l):
            idxs = [traj[k] for traj in ntrajectories]
            nidxs = []
            for _id in idxs:
                if _id is None:
                    nidxs.append(0)
                else:
                    nidxs.append(_id)
            nidxs = torch.tensor(nidxs).to(self._device)
            v = {k: self.buffers[k][nidxs] for k in self.buffers}
            pis = self.position_in_slot[nidxs]
            # Check that slots are full
            if k < max_l - 1:
                for i in range(len(pis)):
                    if not ntrajectories[i][k + 1] is None:
                        assert pis[i] == self.s_slots

            for i in range(len(pis)):
                if not ntrajectories[i][k] is None:
                    length[i] = length[i] + pis[i]

            tensors.append(v)
        ftrajectories = {
            k: torch.cat([t[k] for t in tensors], dim=1) for k in self.buffers
        }
        if erase:
            for k in trajectories:
                for kk in k:
                    if not kk is None:
                        self.set_free_slots(kk)

        return TemporalDictTensor(ftrajectories, length).shorten()
