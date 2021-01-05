#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from __future__ import annotations
import torch
from .dicttensor import DictTensor
from typing import Iterable,Dict,List

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

    def __init__(self, from_dict:Dict[torch.Tensor], lengths:torch.Tensor=None):
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

    def specs(self):
        if self._specs is None:
            s = Specs()
            for k in self.variables:
                s.add(k, self.variables[k][0][0].size(), self.variables.dtype)
            self._specs = s
        return self._specs

    def device(self)->torch.device:
        """
        Returns the device of the TemporalDictTensor
        """
        return self.lengths.device

    def n_elems(self)->int:
        """
        Returns the number of element in the TemporalDictTensor (i.e size of
        the first dimension of each tensor).
        """
        return self.variables[self._keys[0]].size()[0]

    def keys(self)->Iterable[str]:
        """
        Returns the keys in the TemporalDictTensor
        """
        return self.variables.keys()

    def mask(self)->torch.Tensor:
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

    def __getitem__(self, key:str)->torch.Tensor:
        """
        Returns a single tensor of size B x T x ....

        Args:
            key (str): the name of the variable
        """
        return self.variables[key]

    def shorten(self)->TemporalDictTensor:
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

    def unfold(self)->List[TemporalDictTensor]:
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

    def get(self, keys:Iterable[str])->TemporalDictTensor:
        """
        Returns a subset of the TemporalDictTensor depending on the specifed keys

        Args:
            keys (iterable): the keys to keep in the new TemporalDictTensor
        """
        assert not isinstance(keys,str)

        return TemporalDictTensor({k: self.variables[k] for k in keys}, self.lengths)

    def slice(self, index_from:int, index_to:int=None)->TemporalDictTensor:
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

    def temporal_slice(self, index_from:int, index_to:int)->TemporalDictTensor:
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

    def index(self, index:int)->TemporalDictTensor:
        """
        Returns the 1xT TemporalDictTensor for the specified batch index
        """
        v = {k: self.variables[k][index][:] for k in self.variables}
        l = self.lengths[index]
        return TemporalDictTensor(v, l)

    def temporal_index(self, index_t:int)->TemporalDictTensor:
        """
        Return a DictTensor corresponding to the TemporalDictTensor at time
        index_t.
        """
        return DictTensor({k: self.variables[k][:, index_t] for k in self.variables})

    def temporal_multi_index(self, index_t:torch.Tensor)->TemporalDictTensor:
        """
        Return a DictTensor corresponding to the TemporalDictTensor at time index_t
        """
        a=torch.arange(self.n_elems()).to(self.device())
        return DictTensor({k: self.variables[k][a, index_t] for k in self.variables})

    def masked_temporal_index(self, index_t:int)->[DictTensor,torch.Tensor]:
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

    def cat(tensors:Iterable[TemporalDictTensor])->TemporalDictTensor:
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

    def to(self, device:torch.device):
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
        r.append("Length=" + str(self.lengths.numpy()))
        return " ".join(r)

    def __contains__(self, item:str)->bool:
        return item in self.variables

    def expand(self,new_batch_size):        
        """
        Expand a TemporalDictTensor to reach a given batch_size
        """
        assert new_batch_size>self.n_elems()
        diff=new_batch_size-self.n_elems()
        new_lengths=torch.zeros(new_batch_size).long().to(self.device())
        new_lengths[0:self.n_elems()]=self.lengths
        new_variables={}
        
        for k in self.variables.keys():
            s=self.variables[k].size()
            zeros=torch.zeros(diff,*s[1:]).to(self.device())
            nv=torch.cat([self.variables[k],zeros])
            new_variables[k]=nv
        
        return TemporalDictTensor(new_variables,new_lengths)
    
    def copy_(self,source,source_indexes,destination_indexes):
        """
        Copy the values of a source TDT at given indexes to the current TDT at the specified indexes
        """
        assert source_indexes.size()==destination_indexes.size()
        max_length_source=source.lengths.max().item()        
        for k in self.variables.keys():
            self.variables[k][destination_indexes,0:max_length_source]=source[k][source_indexes,0:max_length_source]
        self.lengths[destination_indexes]=source.lengths[source_indexes]
