import torch.nn as nn
import torch 
from mlp_utils import init_weights_normal, init_weights_xavier, sine_init,first_layer_sine_init


class MLPflat(nn.Module):
    """
    A PyTorch implementation of a multi-layer perceptron (MLP) with a flat output layer.
    The MLP consists of a sequence of fully-connected (linear) layers, batch normalization layers, and leaky ReLU activation layers.

    Parameters:
    in_dim (int): Input dimension of the data.
    out_dim (int): Output dimension of the data.
    num_hidden_layers (int): Number of hidden layers in the MLP.
    hidden_features (int): Number of features in each hidden layer.
    initilaization (str): Method of weight initialization for the MLP. Can be 'None', 'normal', 'xavier', 'sine', or 'first_layer'.

    Attributes:
    model (nn.Module): PyTorch Sequential module containing the MLP.
    """

    def __init__(self,in_dim: int, out_dim: int, num_hidden_layers: int, hidden_features: int, initilaization = 'None'):
      super().__init__()
      assert(num_hidden_layers > 0)
      assert(hidden_features > 0)

      net = [nn.Linear(in_dim, hidden_features),nn.BatchNorm1d(hidden_features), nn.LeakyReLU()]
      
      for _ in range(num_hidden_layers): # make N layers
        net += [nn.Linear(hidden_features, hidden_features),nn.BatchNorm1d(hidden_features), nn.LeakyReLU()]
        
      net += [nn.Linear(hidden_features,out_dim,bias=False)]
      
      
      

      self.model = nn.Sequential(*net)

      if initilaization == 'None':
        pass
      else:
        weight_init = {'normal': init_weights_normal,'xavier':init_weights_xavier,'sine':sine_init,'first_layer':first_layer_sine_init}
        self.model.apply(weight_init[initilaization])

      

    def forward(self, x):
      """
        Applies the MLP to the input data.

        Parameters:
        x (torch tensor): Tensor containing the input data.

        Returns:
        output (torch tensor): Tensor containing the output of the MLP.
      """
      x = self.model(x)
      output = x
      return output