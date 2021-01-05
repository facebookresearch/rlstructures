#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""A simple logging interface to display message in the Console
"""

import rlstructures.logging

DEBUG = 0
INFO = 1
NO = 2

__LOGGING_LEVEL = 0


def error(str):
    print("[ERROR] " + str, flush=True)
    assert False


def debug(str):
    global __LOGGING_LEVEL
    if __LOGGING_LEVEL <= 0:
        print("[DEBUG] " + str, flush=True)


def info(str):
    global __LOGGING_LEVEL
    if __LOGGING_LEVEL <= 1:
        print("[INFO] " + str, flush=True)


def basicConfig(**args):
    global __LOGGING_LEVEL
    __LOGGING_LEVEL = args["level"]


def getLogger(str):
    return rlstructures.logging
