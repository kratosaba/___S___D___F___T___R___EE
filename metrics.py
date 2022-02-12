import torch
from mesh import Mesh
import numpy as np 
from grid import Grid
from inside_mesh import check_mesh_contains
def compute_iou(path_gt, path_pr, N=128, sphere=False, sphere_radius=0.25):
    ''' compute iou score
        parameters
            path_gt: path to ground-truth mesh (.ply or .obj)
            path_pr: path to predicted mesh (.ply or .obj)
            N: NxNxN grid resolution at which to compute iou '''

    # load mesh
    occ_pr = Mesh(path_pr) 
    occ_gt = Mesh(path_gt)

    grid = Grid(N,occ_gt.corners).grid
   
    occ_gt = torch.tensor(check_mesh_contains(occ_gt,grid))

    occ_pr = torch.tensor(check_mesh_contains(occ_pr,grid))
    
    # compute iou
    area_union = torch.sum((occ_gt | occ_pr).float())
    area_intersect = torch.sum((occ_gt & occ_pr).float())
    iou = area_intersect / area_union

    return iou