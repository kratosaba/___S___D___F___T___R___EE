import numpy as np
from torch import nn
import torch

def init_weights_normal(m):
    """
    Initializes the weights of a given module using the Kaiming initialization method with a normal distribution.

    Parameters:
    m (nn.Module): PyTorch module to initialize the weights of.
    """
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_weights_xavier(m):
    """
    Initializes the weights of a given module using the Xavier initialization method with a normal distribution.

    Parameters:
    m (nn.Module): PyTorch module to initialize the weights of.
    """
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)

def sine_init(m, w0=30):
    """
    Initializes the weights of a given module using a sine wave initialization method.

    Parameters:
    m (nn.Module): PyTorch module to initialize the weights of.
    w0 (float): Frequency scaling factor for
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor w0
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)