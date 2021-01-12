#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.utils.tensorboard import SummaryWriter
import sqlite3
import os
import os.path
import csv
import copy
from datetime import datetime
import torch
from rlstructures import logging
import numpy as np
import time
import pickle

class Logger:
    """ A logger to store experimental measures in different formats.
    """    
    def __init__(self, **config):
        self.date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.all_values = []
        self.config = config
        self._time = time.time()
        self.keys = {}
        self.keys["training_iteration"] = 1
        self.keys["_clock"] = 1
        for k in self.config:
            self.keys["_hp_" + k] = 1

    def add_images(self, name, value, iteration):
        pass

    def get_scalars(self, name):
        for d in self.all_values:
            r.append(d[name])
        return r

    def _name(self, keys):
        r = "/".join(keys)
        return r

    def _flatten(self, _dict, k=[]):
        r = {}
        for kk in _dict:
            if isinstance(_dict[kk], dict):
                d = self._flatten(_dict[kk], k + [kk])
                r = {**r, **d}
            else:
                r["/".join(k + [kk])] = _dict[kk]
        return r

    def add_dict(self, _dict, iteration):
        d = self._flatten(_dict)
        for k in d:
            self.add_scalar(k, d[k], iteration)

    def add_scalar(self, name, value, iteration):
        if isinstance(value, torch.Tensor):
            value = value.item()

        if len(self.all_values) == 0:
            self.all_values.append({})
            # self.all_values[0]["DATE"] = self.date
            # self.keys["DATE"] = 1
            # self.all_values[0]["FINISHED"] = 0
            # self.keys["FINISHED"] = 1

        while len(self.all_values) <= iteration:
            self.all_values.append({})
        self.all_values[iteration][name] = value
        self.all_values[iteration]["_clock"] = time.time() - self._time
        if not name in self.keys:
            self.keys[name] = 1

    def get_scalars(self, name):
        if isinstance(name, str):
            return [
                self.all_values[i][name]
                for i in range(len(self.all_values))
                if name in self.all_values[i]
            ]
        else:
            name = self._name(name)
            return self.get_scalars(name)

    def copy(self, logger, iteration, from_iteration=-1):
        for k in logger.all_values[from_iteration]:
            self.add_scalar(k, logger.all_values[from_iteration][k], iteration)

    def get_scalar(self, name, iteration=None):
        if iteration is None:
            iteration = len(self.all_values) - 1
        return self.all_values[iteration][name]

    def _to_dict(self, v):
        r = {}
        for k in v:
            kk = k.split("/")
            rr = r
            for i in range(len(kk) - 1):
                kkk = kk[i]
                if not kkk in rr:
                    rr[kkk] = {}
                rr = rr[kkk]
            rr[kk[-1]] = v[k]
        return r

    def get_last(self):
        v = self.all_values[-1]
        return self._to_dict(v)

    def close(self):
        pass


class TFLogger(SummaryWriter, Logger):
    """ A logger that stores informations both in tensorboard and CSV formats
    """    
    def __init__(self, log_dir=None, hps={}, save_every=1):
        SummaryWriter.__init__(self, log_dir=log_dir)
        Logger.__init__(self, **hps)
        self.save_every = save_every
        f = open(log_dir + "/params.json", "wt")
        f.write(str(hps) + "\n")
        f.close()
        self.log_dir=log_dir
        outfile = open(log_dir+"/params.pickle",'wb')
        pickle.dump(hps,outfile)
        outfile.close()

        self.last_csv_update_iteration = 0
        self.csvname = log_dir + "/db.csv"
        self.add_text("Hyperparameters", str(hps))

    def add_images(self, name, value, iteration):
        Logger.add_images(self, name, value, iteration)
        SummaryWriter.add_images(self, name, value, iteration)

    def add_scalar(self, name, value, iteration):
        if not iteration % self.save_every == 0:
            return
        if isinstance(value, int) or isinstance(value, float):
            SummaryWriter.add_scalar(self, name, value, iteration)
        Logger.add_scalar(self, name, value, iteration)

    def update_csv(self):
        length = len(self.all_values)
        values_to_save = self.all_values[self.last_csv_update_iteration : length]

        values = []
        for i in range(len(values_to_save)):
            l = values_to_save[i]
            vv = {**l, "training_iteration": i + self.last_csv_update_iteration}
            vv = {**{"_hp_" + k: self.config[k] for k in self.config}, **vv}
            values.append(vv)

        with open(self.csvname, "a+") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=list(self.keys.keys()))
            if self.last_csv_update_iteration == 0:
                dict_writer.writeheader()
            dict_writer.writerows(values)
        self.last_csv_update_iteration = length

    def close(self):
        SummaryWriter.close(self)
        Logger.close(self)
        self.update_csv()

        f = open(self.log_dir + "/done", "wt")
        f.write("Done\n")
        f.close()
