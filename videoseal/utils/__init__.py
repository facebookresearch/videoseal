# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import time
import sys
import os
import subprocess

import torch

def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


@contextlib.contextmanager
def suppress_output():
    """
    Suppress the print output within a context.
    """
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

def timer_wrapper(func, *args, **kwargs):
    """
    Timer function to measure the execution time of a function.
    """
    start = time.time()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    return result, end - start

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.steps = []

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.steps = []

    def begin(self):
        self.start_time = time.time()
    def start(self):
        self.begin()
    def restart(self):
        self.begin()

    def step(self):
        step_time = time.time() - self.start_time
        self.steps.append(step_time)
    def avg_step(self):
        return sum(self.steps) / len(self.steps)

    def end(self):
        self.end_time = time.time()
        return self.end_time - self.start_time
    def stop(self):
        return self.end()
        