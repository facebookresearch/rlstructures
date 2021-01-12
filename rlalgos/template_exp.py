#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures.logger import Logger, TFLogger
from rlstructures import DictTensor, TemporalDictTensor
from rlstructures import logging
from rlstructures.tools import weight_init
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F

class BaseExperiment:
    def __init__(self, config,create_env,create_agent):
        assert self.check_arguments(config)
        self.config = config
        self.logger = TFLogger(log_dir=self.config["logdir"], hps=self.config)
        self.batchers=[]
        self._create_env=create_env
        self._create_agent=create_agent

    def check_arguments(self,arguments):
        """
        The function aims at checking that the arguments (provided in config) are the good ones
        """
        return True

    def register_batcher(self,batcher):
        """
        Register a new batcher when you create one, to ensure a correct closing of the experiment
        """
        self.batchers.append(batcher)

    def _create_model(self):
        # self.learning_model = ......
        raise NotImplementedError

    def create_model(self):
        self.learning_model = self._create_model()
        self.iteration = 0


    def reset(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def terminate(self):
        for b in self.batchers:
            b.close()
        self.logger.close()

    def go(self):
        self.create_model()
        self.reset()
        self.run()
        self.terminate()
