# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
VideoSeal: Video watermarking library by Facebook AI Research.
"""

__version__ = "1.0"

def load(*args, **kwargs):
    # move the import inside a function, so it is only executed when needed,
    # avoiding import errors during installation.
    from .utils.cfg import setup_model_from_model_card
    return setup_model_from_model_card(*args, **kwargs)
