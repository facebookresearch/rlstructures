#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from __future__ import annotations
import torch
from typing import Iterable, List, Dict


def masked_tensor(tensor0, tensor1, mask):
    """Compute a tensor by combining two tensors with a mask

    :param tensor0: a Bx(N) tensor
    :type tensor0: torch.Tensor
    :param tensor1: a Bx(N) tensor
    :type tensor1: torch.Tensor
    :param mask: a B tensor
    :type mask: torch.Tensor
    :return: (1-m) * tensor 0 + m *tensor1 (averafging is made ine by line)
    :rtype: tensor0.dtype
    """
    s = tensor0.size()
    assert s[0] == mask.size()[0]
    m = mask
    for i in range(len(s) - 1):
        m = mask.unsqueeze(-1)
    m = m.repeat(1, *s[1:])
    m = m.float()
    out = ((1.0 - m) * tensor0 + m * tensor1).type(tensor0.dtype)
    return out


def masked_dicttensor(dicttensor0, dicttensor1, mask):
    """
    Same as `masked_tensor`, but for DictTensor
    """
    variables = {}
    for k in dicttensor0.keys():
        v0 = dicttensor0[k]
        v1 = dicttensor1[k]
        variables[k] = masked_tensor(v0, v1, mask)
    return DictTensor(variables)


class DictTensor:
    """
    A dictionary of torch.Tensor. The first dimension of each tensor is the batch dimension such that all tensors have the same
    batch dimension size.
    """

    def __init__(self, v: Dict = None):
        """Initialize the DictTensor with a dictionary of Tensors.
        All tensors must have the same first dimension size.
        """
        if v is None:
            self.variables = {}
        else:
            self.variables = v
            d = None
            for k in self.variables.values():
                assert isinstance(k, torch.Tensor)
                if d is None:
                    d = k.device
                else:
                    assert d == k.device

    def _check(self) -> bool:
        """
        Check that all tensors have the same batch dimension size.
        """
        s = None
        for v in self.variables.values():
            if s is None:
                s = v.size()[0]
            else:
                assert s == v.size()[0]

    def keys(self) -> Iterable[str]:
        """
        Return the keys of the DictTensor (as an iterator)
        """
        return self.variables.keys()

    def __getitem__(self, key: str) -> torch.Tensor:
        """Get one particular tensor in the DictTensor

        :param key: the name of the tensor
        :type key: str
        :return: the correspondiong tensor
        :rtype: torch.Tensor
        """
        assert key in self.variables,"Key "+key+" not in the DictTensor"
        return self.variables[key]

    def get(self, keys: Iterable[str], clone=False) -> DictTensor:
        """Returns a DictTensor composed of a subset of the tensors specifed by their keys

        :param keys: The keys to keep in the new DictTensor
        :type keys: Iterable[str]
        :param clone: if True, the new DictTensor is composed of clone of the original tensors, defaults to False
        :type clone: bool, optional
        :rtype: DictTensor
        """
        d = DictTensor({k: self.variables[k] for k in keys})
        if clone:
            return d.clone()
        else:
            return d

    def clone(self) -> DictTensor:
        """Clone the dicttensor by cloning all its tensors
        :rtype: DictTensor
        """
        return DictTensor({k: self.variables[k].clone() for k in self.variables})

    def specs(self):
        """
        Return the specifications of the dicttensor as a dictionary
        """
        _specs = {}
        for k in self.variables:
            _specs[k] = {
                "size": self.variables[k][0].size(),
                "dtype": self.variables[k].dtype,
            }
        return _specs

    def device(self) -> torch.device:
        """
        Return the device of the tensors stored in the DictTensor.
        :rtype: torch.device
        """
        if self.empty():
            return None
        return next(iter(self.variables.values())).device

    def n_elems(self) -> int:
        """
        Return the size of size of the batch dimension (i.e the first dimension of the tensors)
        """
        if len(self.variables) > 0:
            f = next(iter(self.variables.values()))
            return f.size()[0]
        # TODO: Empty dicts should be handled better than this
        return 0

    def empty(self) -> bool:
        """Is the DictTensor empty? (no tensors in it)
        :rtype: bool
        """
        return len(self.variables) == 0

    def unfold(self) -> List[DictTensor]:
        """
        Returns a list of DictTensor, each DictTensor capturing one element of the batch dimension (i.e suc that n_elems()==1)
        """
        r = []
        for i in range(self.n_elems()):
            v = {k: self.variables[k][i].unsqueeze(0) for k in self.variables}
            pt = DictTensor(v)
            r.append(pt)
        return r

    def slice(self, index_from: int, index_to: int = None) -> DictTensor:
        """Returns a dict tensor, keeping only batch dimensions between index_from and index_to+1

        :param index_from: The first batch index to keep
        :type index_from: int
        :param index_to: The last+1 batch index to keep. If None, then just index_from is kept
        :type index_to: int, optional
        :rtype: DictTensor
        """
        if not index_to is None:
            v = {}
            for k in self.variables:
                v[k] = self.variables[k][index_from:index_to]
            return DictTensor(v)
        else:
            v = {}
            for k in self.variables:
                v[k] = self.variables[k][index_from]
            return DictTensor(v)

    def index(self, index: int) -> DictTensor:
        """
        The same as self.slice(index)
        """
        v = {k: self.variables[k][index] for k in self.variables}
        return DictTensor(v)

    def cat(tensors: Iterable[DictTensor]) -> DictTensor:
        """
        Aggregate multiple packed tensors over the batch dimension

        Args:
            tensors (list): a list of tensors
        """
        if len(tensors) == 0:
            return DictTensor({})
        retour = {}
        for key in tensors[0].variables:
            to_concat = []
            for n in range(len(tensors)):
                v = tensors[n][key]
                to_concat.append(v)
            retour[key] = torch.cat(to_concat, dim=0)
        return DictTensor(retour)

    def to(self, device: torch.device):
        """
        Create a copy of the DictTensor on a new device (if needed)
        """
        if self.empty():
            return DictTensor({})

        if device == self.device():
            return self.clone()

        v = {}
        for k in self.variables:
            v[k] = self.variables[k].to(device)
        return DictTensor(v)

    def set(self, key: str, value: torch.Tensor):
        """
        Add a tensor to the DictTensor

        Args:
            key (str): the name of the tensor
            value (torch.Tensor): the tensor to add, with a correct batch
                dimension size
        """
        assert self.empty() or value.size()[0] == self.n_elems()
        assert isinstance(value, torch.Tensor)
        assert self.empty() or value.device == self.device()
        self.variables[key] = value

    def unset(self,key: str):
        """ remove one tensor from the dictensot

        Args:
            key (str): the key to remove
        """
        del(self.variables[key])

    def unset_key(self, _str:str):
        """remove all tensors whose key starts with _str
        """
        to_remove=[k for k in self.variables if k.startswith(_str)]
        for k in to_remove:
            self.unset(k)

    def prepend_key(self, _str: str) -> DictTensor:
        """
        Return a new DictTensor where _str has been concatenated to all the keys
        """
        v = {_str + key: self.variables[key] for key in self.variables}
        return DictTensor(v)

    def truncate_key(self, _str: str) -> DictTensor:
        """
        Return a new DictTensor where _str has been removed to all the keys that have _str as a prefix
        """
        v = {}
        for k in self.variables:
            if k.startswith(_str):
                nk = k[len(_str) :]
                v[nk] = self.variables[k]
        return DictTensor(v)

    def __str__(self):
        return "DictTensor: " + str(self.variables)

    def __contains__(self, key: str) -> bool:
        return key in self.variables

    def __add__(self, dt: DictTensor) -> DictTensor:
        """
        Create a new DictTensor containing all the tensors from self and dt
        """
        if self.empty():
            return dt.clone()
        if dt.empty():
            return self.clone()
        assert dt.device() == self.device()
        for k in dt.keys():
            assert not k in self.variables, (
                "variable " + k + " already exists in the DictTensor"
            )
        v = {**self.variables, **dt}
        return DictTensor(v)

    def copy_(self, source, source_indexes, destination_indexes):
        """
        Copy the values of a source TDT at given indexes to the current TDT at the specified indexes
        """
        assert source_indexes.size() == destination_indexes.size()
        for k in self.variables.keys():
            self.variables[k][destination_indexes] = source[k][source_indexes]


class TemporalDictTensor:
    """
    Describe a batch of temporal tensors where:
    * each tensor has a name
    * each tensor is of size B x T x ...., where B is the batch index, and T
        the time index
    * the length tensor gives the number of timesteps for each batch

    It is an extension of DictTensor where a temporal dimension has been added.
    The structure also allows dealing with batches of sequences of different
    sizes.

    Note that self.lengths returns a tensor of the lengths of each element of
    the batch.
    """

    def __init__(self, from_dict: Dict[torch.Tensor], lengths: torch.Tensor = None):
        """
        Args:
            from_dict (dict of tensors): the tensors to store.
            lengths (long tensor): the length of each element in the batch. If
                None, then use the second dimension of the tensors to compute
                the length.
        """
        self.variables = from_dict
        self._keys = list(self.variables.keys())

        self.lengths = lengths
        if self.lengths is None:
            self.lengths = (
                torch.ones(self.n_elems()).long()
                * self.variables[self._keys[0]].size()[1]
            )
        assert self.lengths.dtype == torch.int64
        self._specs = None

    def clone(self):
        v = {k: self.variables[k].clone() for k in self.variables}
        return TemporalDictTensor(v, lengths=self.lengths.clone())

    def set(self, name, tensor):
        self.variables[name] = tensor

    def specs(self):
        if self._specs is None:
            s = Specs()
            for k in self.variables:
                s.add(k, self.variables[k][0][0].size(), self.variables.dtype)
            self._specs = s
        return self._specs

    def device(self) -> torch.device:
        """
        Returns the device of the TemporalDictTensor
        """
        return self.lengths.device

    def n_elems(self) -> int:
        """
        Returns the number of element in the TemporalDictTensor (i.e size of
        the first dimension of each tensor).
        """
        return self.variables[self._keys[0]].size()[0]

    def keys(self) -> Iterable[str]:
        """
        Returns the keys in the TemporalDictTensor
        """
        return self.variables.keys()

    def mask(self) -> torch.Tensor:
        """
        Returns a mask over sequences based on the length of each trajectory

        Considering that the TemporalDictTensor is of size B x T, the mask is
        a float tensor (0.0 or 1.0) of size BxT. A 0.0 value means that the
        value at b x t is not set in the TemporalDictTensor.
        """
        for k in self.variables:
            max_length = self.variables[k].size()[1]
            _mask = (
                torch.arange(max_length)
                .to(self.lengths.device)
                .unsqueeze(0)
                .repeat(self.n_elems(), 1)
            )
            _mask = _mask.lt(self.lengths.unsqueeze(1).repeat(1, max_length)).float()
            return _mask

    def __getitem__(self, key: str) -> torch.Tensor:
        """
        Returns a single tensor of size B x T x ....

        Args:
            key (str): the name of the variable
        """
        return self.variables[key]

    def shorten(self) -> TemporalDictTensor:
        """
        Restrict the size of the variables (in term of timesteps) to provide the smallest
        possible tensors.

        If the TemporalDictTensor is of size B x T, considering that
        Tmax = self.lengths.max(), then it returns a TemporalDictTensor of size B x Tmax
        """
        ml = self.lengths.max()
        v = {k: self.variables[k][:, :ml] for k in self.variables}
        pt = TemporalDictTensor(v, self.lengths.clone())
        return pt

    def unfold(self) -> List[TemporalDictTensor]:
        """
        Return a list of TemporalDictTensor of size 1 x T
        """
        r = []
        for i in range(self.n_elems()):
            v = {k: self.variables[k][i].unsqueeze(0) for k in self.variables}
            l = self.lengths[i].unsqueeze(0)
            pt = TemporalDictTensor(v, l)
            r.append(pt)
        return r

    def get(self, keys: Iterable[str]) -> TemporalDictTensor:
        """
        Returns a subset of the TemporalDictTensor depending on the specifed keys

        Args:
            keys (iterable): the keys to keep in the new TemporalDictTensor
        """
        assert not isinstance(keys, str)

        return TemporalDictTensor({k: self.variables[k] for k in keys}, self.lengths)

    def slice(self, index_from: int, index_to: int = None) -> TemporalDictTensor:
        """
        Returns a slice (in the batch dimension)
        """
        if not index_to is None:
            v = {k: self.variables[k][index_from:index_to] for k in self.variables}
            l = self.lengths[index_from:index_to]
            return TemporalDictTensor(v, l)
        else:
            v = {k: self.variables[k][index_from].unsqueeze(0) for k in self.variables}
            l = self.lengths[index_from].unsqueeze(0)
            return TemporalDictTensor(v, l)

    def temporal_slice(self, index_from: int, index_to: int) -> TemporalDictTensor:
        """
        Returns a slice (in the temporal dimension)
        """
        v = {k: self.variables[k][:, index_from:index_to] for k in self.variables}

        # Compute new length
        l = self.lengths - index_from
        l = torch.clamp(l, 0)

        m = torch.ones(*l.size()) * (index_to - index_from)
        m = m.to(self.device())
        low = l.lt(m).float()
        m = low * l + (1 - low) * m
        return TemporalDictTensor(v, m.long())

    def index(self, index: int) -> TemporalDictTensor:
        """
        Returns the 1xT TemporalDictTensor for the specified batch index
        """
        v = {k: self.variables[k][index][:] for k in self.variables}
        l = self.lengths[index]
        return TemporalDictTensor(v, l)

    def temporal_index(self, index_t: int) -> TemporalDictTensor:
        """
        Return a DictTensor corresponding to the TemporalDictTensor at time
        index_t.
        """
        return DictTensor({k: self.variables[k][:, index_t] for k in self.variables})

    def temporal_multi_index(self, index_t: torch.Tensor) -> TemporalDictTensor:
        """
        Return a DictTensor corresponding to the TemporalDictTensor at time index_t
        """
        a = torch.arange(self.n_elems()).to(self.device())
        return DictTensor({k: self.variables[k][a, index_t] for k in self.variables})

    def masked_temporal_index(self, index_t: int) -> [DictTensor, torch.Tensor]:
        """
        Return a DictTensor at time t along with a mapping vector
        Considering the TemporalDictTensor is of size BxT, the method returns
        a TemporalDictTensor of size B'xT and a tensor of size B' where:
            * only the B' relevant dimension has been kept (depending on the
                index_t < self.lengths criterion)
            * the mapping vector maps each of the B' dimension to the B
                dimension of the original TemporalDictTensor
        """
        m = torch.tensor([index_t]).repeat(self.n_elems())
        m = m.lt(self.lengths)
        v = {k: self.variables[k][m, index_t] for k in self.variables}
        m = torch.arange(self.n_elems())[m]
        return DictTensor(v), m

    def cat(tensors: Iterable[TemporalDictTensor]) -> TemporalDictTensor:
        """
        Aggregate multiple packed tensors over the batch dimension

        Args:
            tensors (list): a list of tensors
        """
        lengths = torch.cat([t.lengths for t in tensors])
        lm = lengths.max().item()
        retour = {}
        for key in tensors[0].keys():
            to_concat = []
            for n in range(len(tensors)):
                v = tensors[n][key]
                s = v.size()
                s = (s[0],) + (lm - s[1],) + s[2:]
                if s[1] > 0:
                    toadd = torch.zeros(s, dtype=v.dtype)
                    v = torch.cat([v, toadd], dim=1)
                to_concat.append(v)
            retour[key] = torch.cat(to_concat, dim=0)
        return TemporalDictTensor(retour, lengths)

    def to(self, device: torch.device):
        """
        Returns a copy of the TemporalDictTensor to the provided device (if
        needed).
        """
        if device == self.device():
            return self

        lengths = self.lengths.to(device)
        v = {}
        for k in self.variables:
            v[k] = self.variables[k].to(device)
        return TemporalDictTensor(v, lengths)

    def __str__(self):
        r = ["TemporalDictTensor:"]
        for k in self.variables:
            r.append(k + ":" + str(self.variables[k].size()))
        r.append("Lengths =" + str(self.lengths.numpy()))
        return " ".join(r)

    def __contains__(self, item: str) -> bool:
        return item in self.variables

    def full(self):
        """returns True if self.lengths==self.lengts.max() => No empty element"""
        return self.mask().sum() == 0.0

    def expand(self, new_batch_size):
        """
        Expand a TemporalDictTensor to reach a given batch_size
        """
        assert new_batch_size > self.n_elems()
        diff = new_batch_size - self.n_elems()
        new_lengths = torch.zeros(new_batch_size).long().to(self.device())
        new_lengths[0 : self.n_elems()] = self.lengths
        new_variables = {}

        for k in self.variables.keys():
            s = self.variables[k].size()
            zeros = torch.zeros(diff, *s[1:]).to(self.device())
            nv = torch.cat([self.variables[k], zeros])
            new_variables[k] = nv

        return TemporalDictTensor(new_variables, new_lengths)

    def copy_(self, source, source_indexes, destination_indexes):
        """
        Copy the values of a source TDT at given indexes to the current TDT at the specified indexes
        """
        assert source_indexes.size() == destination_indexes.size()
        max_length_source = source.lengths.max().item()
        for k in self.variables.keys():
            self.variables[k][destination_indexes, 0:max_length_source] = source[k][
                source_indexes, 0:max_length_source
            ]
        self.lengths[destination_indexes] = source.lengths[source_indexes]


class Trajectories:
    def __init__(self, info, trajectories):
        self.info = info
        self.trajectories = trajectories
        assert info.empty() or self.info.device() == self.trajectories.device()
        assert self.info.empty() or self.info.n_elems() == self.trajectories.n_elems()

    def to(self, device):
        return Trajectories(self.info.to(device), self.trajectories.to(device))

    def device(self):
        return self.trajectories.device()

    def cat(trajectories: Iterable[Trajectories]):
        return Trajectories(
            DictTensor.cat([t.info for t in trajectories]),
            TemporalDictTensor([t.trajectories for t in trajectories]),
        )

    def n_elems(self):
        return self.trajectories.n_elems()

    def sample(self, n):
        raise NotImplementedError
