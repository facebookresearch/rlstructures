#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from __future__ import annotations
import torch
from typing import Iterable,List,Dict

def masked_tensor(tensor0,tensor1,mask):
    """  Compute a tensor by combining two tensors with a mask

    :param tensor0: a Bx(N) tensor
    :type tensor0: torch.Tensor
    :param tensor1: a Bx(N) tensor
    :type tensor1: torch.Tensor
    :param mask: a B tensor
    :type mask: torch.Tensor
    :return: (1-m) * tensor 0 + m *tensor1 (averafging is made ine by line)
    :rtype: tensor0.dtype
    """
    s=tensor0.size()
    assert s[0]==mask.size()[0]
    m=mask
    for i in range(len(s)-1):
        m=mask.unsqueeze(-1)
    m=m.repeat(1,*s[1:])
    m=m.float()
    out=((1.0-m)*tensor0+m*tensor1).type(tensor0.dtype)
    return out

def masked_dicttensor(dicttensor0,dicttensor1,mask):
    """
    Same as `masked_tensor`, but for DictTensor
    """
    variables={}
    for k in dicttensor0.keys():
        v0=dicttensor0[k]
        v1=dicttensor1[k]
        variables[k]=masked_tensor(v0,v1,mask)
    return DictTensor(variables)

class DictTensor:
    """
    A dictionary of torch.Tensor. The first dimension of each tensor is the batch dimension such that all tensors have the same
    batch dimension size.
    """
    def __init__(self, v:Dict=None):
        """ Initialize the DictTensor with a dictionary of Tensors.
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

    def _check(self)->bool:
        """
        Check that all tensors have the same batch dimension size.
        """
        s = None
        for v in self.variables.values():
            if s is None:
                s = v.size()[0]
            else:
                assert s == v.size()[0]

    # def _check_specs(self, specs)->bool:
    #     """ Check that a DictTensor follow a particular specification

    #     :param specs: specs (dict): a dictionary where each key is associated with
    #                     'dtype' and 'size' values
    #     :return: True or False
    #     :rtype: bool
    #     """
    #     for k in specs:
    #         assert k in self.variables, "Variable not found"

    def keys(self)-> Iterable[str]:
        """
        Return the keys of the DictTensor (as an iterator)
        """
        return self.variables.keys()

    def __getitem__(self, key:str)->torch.Tensor:
        """Get one particular tensor in the DictTensor

        :param key: the name of the tensor
        :type key: str
        :return: the correspondiong tensor
        :rtype: torch.Tensor
        """
        return self.variables[key]

    def get(self, keys:Iterable[str],clone=False)-> DictTensor:
        """ Returns a DictTensor composed of a subset of the tensors specifed by their keys

        :param keys: The keys to keep in the new DictTensor
        :type keys: Iterable[str]
        :param clone: if True, the new DictTensor is composed of clone of the original tensors, defaults to False
        :type clone: bool, optional
        :rtype: DictTensor
        """
        d=DictTensor({k: self.variables[k] for k in keys})
        if clone:
            return d.clone()
        else:
            return d

    def clone(self)->DictTensor:
        """Clone the dicttensor by cloning all its tensors
        :rtype: DictTensor
        """
        return DictTensor({k:self.variables[k].clone() for k in self.variables})

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

    def device(self)->torch.device:
        """
        Return the device of the tensors stored in the DictTensor.
        :rtype: torch.device
        """
        return next(iter(self.variables.values())).device

    def n_elems(self)->int:
        """
        Return the size of size of the batch dimension (i.e the first dimension of the tensors)
        """
        if len(self.variables) > 0:
            f = next(iter(self.variables.values()))
            return f.size()[0]
        # TODO: Empty dicts should be handled better than this
        return 0

    def empty(self)->bool:
        """ Is the DictTensor empty? (no tensors in it)
        :rtype: bool
        """
        return len(self.variables)==0

    def unfold(self)->List[DictTensor]:
        """
        Returns a list of DictTensor, each DictTensor capturing one element of the batch dimension (i.e suc that n_elems()==1)
        """
        r = []
        for i in range(self.n_elems()):
            v = {k: self.variables[k][i].unsqueeze(0) for k in self.variables}
            pt = DictTensor(v)
            r.append(pt)
        return r

    def slice(self, index_from:int, index_to:int=None)->DictTensor:
        """ Returns a dict tensor, keeping only batch dimensions between index_from and index_to+1

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

    def index(self, index:int)->DictTensor:
        """
        The same as self.slice(index)
        """
        v = {k: self.variables[k][index] for k in self.variables}
        return DictTensor(v)

    def cat(tensors:Iterable[DictTensor])->DictTensor:
        """
        Aggregate multiple packed tensors over the batch dimension

        Args:
            tensors (list): a list of tensors
        """
        if (len(tensors)==0):
            return DictTensor({})
        retour = {}
        for key in tensors[0].variables:
            to_concat = []
            for n in range(len(tensors)):
                v = tensors[n][key]
                to_concat.append(v)
            retour[key] = torch.cat(to_concat, dim=0)
        return DictTensor(retour)

    def to(self, device:torch.device):
        """
        Create a copy of the DictTensor on a new device (if needed)
        """
        if device == self.device():
            return self

        v = {}
        for k in self.variables:
            v[k] = self.variables[k].to(device)
        return DictTensor(v)

    def set(self, key:str, value:torch.Tensor):
        """
        Add a tensor to the DictTensor

        Args:
            key (str): the name of the tensor
            value (torch.Tensor): the tensor to add, with a correct batch
                dimension size
        """
        assert value.size()[0] == self.n_elems()
        assert isinstance(value, torch.Tensor)
        assert value.device==self.device()
        self.variables[key] = value

    def prepend_key(self, _str:str)->DictTensor:
        """
        Return a new DictTensor where _str has been concatenated to all the keys
        """
        v = {_str + key: self.variables[key] for key in self.variables}
        return DictTensor(v)

    def truncate_key(self, _str:str)->DictTensor:
        """
        Return a new DictTensor where _str has been removed to all the keys that have _str as a prefix
        """
        v = {}
        for k in self.variables:
            if k.startswith(_str):
                nk=k[len(_str):]
                v[nk]=self.variables[k]
        return DictTensor(v)

    def __str__(self):
        return "DictTensor: " + str(self.variables)

    def __contains__(self, key:str)->bool:
        return key in self.variables

    def __add__(self, dt:DictTensor)->DictTensor:
        """
        Create a new DictTensor containing all the tensors from self and dt
        """
        assert dt.device()==self.device()
        for k in dt.keys():
            assert not k in self.variables, (
                "variable " + k + " already exists in the DictTensor"
            )
        v = {**self.variables, **dt}
        return DictTensor(v)
