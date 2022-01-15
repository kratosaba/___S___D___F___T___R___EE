from mesh_to_sdf import *

import trimesh
import numpy as np
from skimage import measure
from math import pi

import torch.nn as nn

from sample_utlis import get_normal_batch
from meshing_utils import recenter_mesh, as_mesh 

class Mesh:
    """
    Instantiates mesh class, that have upload, sample and calculate sdf method
    """
    
    def __init__(self,name):
        self.name = name
        self.mesh = self.upload()
        self.corners = [self.mesh.vertices.min(0) - 1e-2, self.mesh.vertices.max(0) + 1e-2]
        

    
    def upload(self):
        mesh = trimesh.load(self.name)
        mesh = as_mesh(mesh)
        recenter_mesh(mesh)
        return mesh


    def sample(self,sample_size,perc = 0.02):
        c0, c1 = self.corners
        random_sammples = np.random.uniform(size=[int(sample_size*perc), 3]) * (c1-c0) + c0
        sample_normal, batch_normals = get_normal_batch(self.mesh, sample_size)
        sample_normal_perturbed = sample_normal + np.random.normal(size=[int(sample_size),3]) * .01
        xyz = np.concatenate((sample_normal,random_sammples),axis=0)
        xyz = np.concatenate((xyz,sample_normal_perturbed),axis=0)
        return xyz
        
    
    def calculate_sdf(self,samples):
        sdf = mesh_to_sdf(self.mesh,samples)
        samples_sdf = np.concatenate((samples,sdf.reshape(sdf.shape[0],1)),1)
        return samples_sdf


    def reconstruct(self,implicit_sdf,dimensions):
        
        c0, c1 = self.corners
        spacingx,spacingy,spacingz = abs((c1[0])-(c0[0]))/dimensions,abs((c1[1])-(c0[1]))/dimensions,abs((c1[2])-(c0[2]))/dimensions

        
        implicit_sdf = implicit_sdf.reshape((np.cbrt(implicit_sdf.shape[0]).astype(np.int32),np.cbrt(implicit_sdf.shape[0]).astype(np.int32),np.cbrt(implicit_sdf.shape[0]).astype(np.int32)))
        if not isinstance(implicit_sdf, np.ndarray) :
            implicit_sdf = implicit_sdf.detach().cpu().numpy()  
        
        iso_val= 0.005

        verts, faces,normals,values = measure.marching_cubes(implicit_sdf ,iso_val,spacing=(spacingz,spacingx,spacingy))

        
        rec_mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_normals=normals) 
        
        return rec_mesh