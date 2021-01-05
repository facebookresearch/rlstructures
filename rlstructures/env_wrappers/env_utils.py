#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import numpy as np


def format_frame(frame):
    if isinstance(frame, dict):
        r = {}
        for k in frame:
            r[k] = format_frame(frame[k])
        return r
    elif isinstance(frame, list):
        t=torch.tensor(frame).unsqueeze(0)
        if t.dtype==torch.float64:
            t=t.float()
        return t
    elif isinstance(frame, np.ndarray):
        t=torch.from_numpy(frame).unsqueeze(0)
        if t.dtype==torch.float64:
            t=t.float()
        return t
    elif isinstance(frame, torch.Tensor):
        return frame.unsqueeze(0) #.float()
    elif isinstance(frame, int):    
        return torch.tensor([frame]).unsqueeze(0).long()
    elif isinstance(frame, float):
        return torch.tensor([frame]).unsqueeze(0).float()
        return t

    else:
        try:
            # Check if its a LazyFrame from OpenAI Baselines
            o = torch.from_numpy(frame._force()).unsqueeze(0).float()
            return o
        except:
            assert False
