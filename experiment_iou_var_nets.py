"""
Experiment main objectives:
    - Check the iou metric over different depths.
    - Define whether the tree should be uniform, i.e. every network in each child be the same, or to be variable over each child.
    
For this we need to:
    - Save all the iou and the training time per reconstruction, with different hyper parameters so we can make a line chart to see the numbers and define our parameters.
    - Then using choose the hyper parameters to choose the variable and uniform trees.  

"""

"""
 Args and hyper parameters for training:
 maxdepth: maximum depth
 inTrain samples: input training set
 epochs
 batchsize
 errorTolerance
 N: number of hidden layers
 H: 'Hieght' of layer
 porcentage: porcentage for importance sampling
 weightdecay: weightdecay
 importance_sampling: If you would like to do the sampling

 Measuring the performance of the experiment:
 - Time it takes to train.
 - IOU measure.
 - Quality of visual recontruction. 
 """


import metrics
import time
from mesh import Mesh
import numpy as np
from tree import Tree
from grid import Grid
from utils import save_experiment

import numpy as np
import random
import torch
from sklearn.model_selection import ParameterGrid

#setting gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


param_grid = {
    'k':[0],
    'max_depth':[1,2,3,4],
    'train_samples':[2**16,2**18,2**20],
    'epochs':[50,100], 
    'in_dimesion': [3],
    'errorTolerance':[0.01,0.001],
    'num_hidden_layers':[1,2,4,5,6,7],
    'hidden_features': [32,64,128],
    'percentage_used_trained_importance': [0.25,0.5,0.75],
    'weightdecay':[1e-6],
    'importance_sampling': [True,False],
    'percentage_random_sample': [0.05,0.1,0.15],
    'bartch_per':[0.0024, 0.02,0.2]
    }

# Creates a param grid and randomly chooses 100 samples to be used    
param_list =  np.array(ParameterGrid(param_grid)) # can save into text file.
random_grid_list = random.sample(range(0, len(param_list)), 100)
mesh = Mesh('./models/dragon_final.ply')


#loop through random grid list
for grid in param_list[random_grid_list]:
    num_experiment = 0

    # sample around the mesh
    sample = mesh.sample(grid['train_samples'],grid['percentage_random_sample'])
    sdf_sample = mesh.calculate_sdf(sample)
    
    #Sepparate parameters to use for train function
    train_param = {key:grid[key] for key in grid if key!='percentage_random_sample' and key!= 'train_samples' and key!= 'batch_per'}
    train_param['train_samples'] = sdf_sample
    train_param['batchsize'] = int(sdf_sample.shape[0]*grid['batch_per'] )
    
    # create empty tree 
    mesh_tree = Tree()
    
    #train the tree
    start = time.process_time()
    mesh_tree.train(**train_param)
    train_time =   time.process_time() - start  
    
    #sample space in a uniform grid
    uniform_grid = Grid(150,mesh.corners)
    
    # evaluate the sdf tree
    grid_sdf = mesh_tree.evaluate(uniform_grid.grid) 
    
    # create a mesh reconstructed
    mesh_reconstructed = mesh.reconstruct(grid_sdf,150)
    mesh_reconstructed.export(f'./reconstructions/dragon_reconstruction{num_experiment}')

    # calculate iou
    score = metrics.compute_iou('./models/dragon_final.ply',f'./reconstructions/dragon_reconstruction{num_experiment}')

    # final dictionary to save as paramenters used and results
    final_parameters = grid
    final_parameters['train_time'] = train_time
    final_parameters['iou_score'] = score
    
    # Save score and metrics used for the num_experiment
    save_experiment(f'./Parameter_and_results/dragon_reconstruction_{num_experiment}_paramenters_results.txt',final_parameters)

    num_experiment+=1