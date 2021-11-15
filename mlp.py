import torch.nn as nn
import torch 
from mlp_utils import init_weights_normal, init_weights_xavier, sine_init,first_layer_sine_init


class MLPflat(nn.Module):
    """
    Initialises the neural network, and defines the method foward
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
        weights = weight_init[initilaization]
        self.model.apply(weights)

      

    def forward(self, x):
      x = self.model(x)
      output = x
      return output