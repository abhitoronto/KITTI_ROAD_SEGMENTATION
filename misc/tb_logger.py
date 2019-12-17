"""
    This module contains the auxilary class which allows make logging in TensorBoardX
"""

import tensorflow as tf
import numpy as np
import torch, os
import torch.nn as nn
from tensorboardX import SummaryWriter

class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, step):
        """
            Log a scalar variable.
            params:
                tag     : Name of the scalar
                value   : value itself
                step    :  training iteration
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, values: dict, step):
        """
            Log scalar variables.
            params:
                same as log_scalar
        """
        self.writer.add_scalars(tag, values, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        raise NotImplementedError("This method is not implemented.")

    def log_image(self, img_tensor, epoch: int, dataformats='CHW' ,title: str='image'):
        """
            Log an image.
            params:
        """
        self.writer.add_image(title, img_tensor, epoch, dataformats=dataformats)
