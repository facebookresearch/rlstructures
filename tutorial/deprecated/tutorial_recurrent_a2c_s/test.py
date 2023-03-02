#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn

l = nn.Sequential(nn.Linear(64, 32))

while True:
    a = torch.randn(1024, 64, dtype=torch.float32)
    y = l(a)
    yy = l(a[:10])
    print((y[:10] - yy).sum())
