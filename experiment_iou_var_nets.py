"""
Experiment main objectives:
    - Check the iou metric over different depths, over network hyper parameters
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
import torch
from sklearn.model_selection import ParameterGrid
import math
from trimesh import  transformations
import meshing_utils

def main():
    #setting gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    param_grid = {
        'epochs':[50,100],
        'hidden_features': [32,64,128],
        'num_hidden_layers':[1,2,3,4], # changes this one first as well alphabetical that can 
        'max_depth':[1,2,3],
        'k':[0],
        'in_dimesion': [3],
        'weightdecay':[1e-6],
        'train_samples':[2**18],
        'errorTolerance':[0.001],
        'importance_sampling': [False],
        'percentage_used_trained_importance': [0.75], # random choosen to do a small experiment sinse is without importance sampling
        'bartch_per':[0.0024],
        'percentage_random_sample': [0.1], # 10% per choosen randomly
        }

    # Creates a param grid and randomly chooses 100 samples to be used    
    param_list =  np.array(ParameterGrid(param_grid)) # can save into text file.
    mesh = Mesh('./models/dragon_final.ply')

    num_experiment = 0

    #loop through random grid list
    for grid in param_list:
        print("Experiment: ", num_experiment)

        # sample around the mesh
        sample = mesh.sample(grid['train_samples'],grid['percentage_random_sample'])
        sdf_sample = mesh.calculate_sdf(sample)
        
        #Sepparate parameters to use for train function
        train_param = {key:grid[key] for key in grid if key!='percentage_random_sample' and key!= 'train_samples' and key!= 'batch_per'}
        train_param['train_samples'] = torch.from_numpy(sdf_sample.astype(np.float32)).to(device)
        train_param['batchsize'] = int(sdf_sample.shape[0]*grid['batch_per'] )
        
        # create empty tree 
        mesh_tree = Tree()
        
        #train the tree
        start = time.process_time()
        mesh_tree.train(**train_param)
        train_time =   time.process_time() - start  
        print("Train time: ",train_time, ", Experiment: ", num_experiment)
        #sample space in a uniform grid
        uniform_grid = Grid(150,mesh.corners)

        # evaluate the sdf tree
        grid_sdf = mesh_tree.evaluate(torch.from_numpy(uniform_grid.grid.astype(np.float32)).to(device)) 
        
        # create a mesh reconstructed
        
        mesh_reconstructed = mesh.reconstruct(grid_sdf,150)
        
        # TODO create a normalazation function that works 
        angle = math.pi / 2
        direction_1 = [0, -1, 0]
        direction_2 = [0, 0, -1]
        center = [0, 0, 0]

        rot_matrix_1 = transformations.rotation_matrix(angle, direction_1, center)
        rot_matrix_2 = transformations.rotation_matrix(angle, direction_2, center)


        mesh_reconstructed.apply_transform(rot_matrix_1)
        mesh_reconstructed.apply_transform(rot_matrix_2)


        meshing_utils.recenter_mesh(mesh_reconstructed)

        # Saving the reconstruction for comparation to the reference mesh
        mesh_reconstructed.export(f'./reconstructions/dragon_reconstruction{num_experiment}.stl')
        
        # calculate iou
        score = metrics.compute_iou('./models/dragon_final.ply',f'./reconstructions/dragon_reconstruction{num_experiment}.stl')

        # final dictionary to save as paramenters used and results
        final_parameters = grid
        final_parameters['train_time'] = train_time
        final_parameters['iou_score'] = score.item()
        
        # Save score and metrics used for the num_experiment
        save_experiment(f'./Parameter_and_results/dragon_reconstruction_{num_experiment}_paramenters_results.txt',final_parameters)

        num_experiment+=1