import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Grid:
    def __init__(self,dimensions,corners):
        self.dimensions = dimensions
        self.corners = corners
        
        c0, c1 = self.corners
        x = np.arange((c0[0]),(c1[0]),abs((c1[0])-(c0[0]))/dimensions)
        y= np.arange((c0[1]),(c1[1]),abs((c1[1])-(c0[1]))/dimensions)
        z = np.arange((c0[2]),(c1[2]),abs((c1[2])-(c0[2]))/dimensions)
        

        xval, yval, zval = np.meshgrid(x,y,z)
        grid = np.array((xval,yval,zval)).T.reshape(-1,3)
        
        self.grid = grid