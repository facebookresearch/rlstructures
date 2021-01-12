#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


###### DictTensor

# A DictTensor is dictionary of pytorch tensors. It assumes that the 
# first dimension of each tensor contained in the DictTensor is the batch dimension
# The easiest way to build a DictTensor is to use a ditcionnary of tensors as input

from rlstructures import DictTensor
import torch
d=DictTensor({"x":torch.randn(3,5),"y":torch.randn(3,8)})

# The number of batches is accessible through n_elems()
print(d.n_elems()," <- number of elements in the batch")


# Many methods can be used over DictTensor (see DictTensor documentation)

d["x"] # Returns the tensor 'x' in the DictTensor
d.keys() # Returns the names of the variables of the DictTensor

# An empty DictTensor can be defined as follows:
d=DictTensor({})



###### TemporalDictTensor

# A TemporalDictTensor is a sequence of DictTensors. In memory, it is stored as a dictionary of tensors, 
# where the first dimesion is the batch dimension, and the second dimension is the time index. 
# Each element in the batch is a sequence, and two sequences can have a different length.etc...")

from rlstructures import TemporalDictTensor

#Create three sequences of variables x and y, where the length of the first sequence is 6, the length of the second is 10  and the length of the last sequence is 3
d=TemporalDictTensor({"x":torch.randn(3,10,5),"y":torch.randn(3,10,8)},lengths=torch.tensor([6,10,3]))

print(d.n_elems()," <- number of elements in the batch")
print(d.lengths,"<- Lengths of the sequences")
print(d["x"].size(),"<- access to the tensor 'x'")

print("Masking: ")
print(d.mask())

print("Slicing (restricting the sequence to some particular temporal indexes) ")
d_slice=d.temporal_slice(0,4)
print(d_slice.lengths)
print(d_slice.mask())

# IMPORTANT: DictTensor and TemporalDictTensor can be moved to cpu/gpu using the .to method
