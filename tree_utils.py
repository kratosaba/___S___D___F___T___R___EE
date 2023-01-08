import torch
from scipy.sparse import coo_matrix , dia_matrix, lil_matrix
from sklearn.neighbors import BallTree
import numpy as np 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linearfunction(X,m,b):
  """
  This defines a function called linearfunction that takes in three arguments: X, m, and b. The function returns the result of m * X + b
  """
  return m*X+b
  
def split_into_quadrants(points,tree, idx=0):
    """
    This defines a function called split_into_quadrants that takes in three arguments: points, tree, and idx. The function returns a list of points split into quadrants.
    """

    if idx < points.shape[-1]-1:
        #median = torch.median(points[:,idx])
        
        # This next code block I'm not sure about, how to how to find the exact value of the right value and the left value
        right =  torch.min(points[:,idx]) + (torch.max(points[:,idx]) - torch.min(points[:,idx]))*0.60 #question
        left =   torch.min(points[:,idx])+(torch.max(points[:,idx]) - torch.min(points[:,idx])) *0.40   
        
        
        if (tree.rigth_cut_off_values[idx]==1) and (tree.left_cut_off_values[idx]==1):
          tree.rigth_cut_off_values[idx]= right # question
          tree.left_cut_off_values[idx]= left
        
        positive= points[points[:, idx] <= right]
        negative= points[points[:, idx] >= left]
        
        return (split_into_quadrants(positive,tree, idx+1) +
                split_into_quadrants(negative,tree, idx+1))
    else:
      return [points]

def split_val_quadrants(domain,tree, idnx=0):
    # This function splits the given domain into quadrants based on the cut-off values in the provided tree and the given index.
    # The domain is recursively split until the index reaches the end of the domain's dimensions.
    # Returns a list of quadrants.

    if idnx < (domain.shape[-1]):
        right = tree.rigth_cut_off_values[idnx]
        left = tree.left_cut_off_values[idnx]

        positive= domain[domain[:, idnx] <= right]
        negative= domain[domain[:, idnx] >= left]
        #print(len(positive),len(negative))
        if len(positive)==0 or len(negative)==0:
          print(right,left)
        return (split_val_quadrants(positive,tree, idnx+1) +
                split_val_quadrants(negative,tree, idnx+1))
    else:
      
      return [domain]

def train_network(phi,train_loader,epochs,criterion,optimizer,depth):
    # This function trains the given network (phi) using the provided train data (train_loader), number of epochs, loss criterion, optimizer, and depth.
    # For each epoch, the function loops through the batch of training data, optimizes the network's parameters, and calculates the loss.

    for epoch in range(epochs):
        batch = 0
        for x_batch, y_batch in train_loader:
            if len(x_batch) == 1:
              continue
            else:
              optimizer.zero_grad()
              
              
              y_pred = phi(x_batch.to(device))
              
          
              loss = criterion(y_pred.squeeze(), y_batch.to(device).squeeze())

              
              loss.backward(retain_graph=True)
              optimizer.step() # Optimizes only phi parameters
              batch+=1

def laplace_operator(Nsamples,r_nn,tol=10e-6):
  # This function calculates the Laplacian operator for the given samples using a ball tree and tolerance value.
  # It returns the Laplacian matrix.
  knn = BallTree(Nsamples,leaf_size = 5) 
  A=lil_matrix((Nsamples.shape[0],Nsamples.shape[0]) )

  for i in range(0,A.shape[0]):
    dist, ind = knn.query([Nsamples[i]], k=r_nn) # get the r_nn clossest neightbors 
    squared_sigma = -(dist[dist == np.max(dist)])**2 / 2*(np.log(tol)) # question 
    for j in ind[0,1:]:
      number = -np.float128(dist[ind==j]**2)/2*(np.float128(squared_sigma))
      weight = np.exp(np.float128(number))
      if len(weight) >1:
        weight = weight[0]
      A[i,j] = weight
  
  A = (A+A.transpose())/2
  D = coo_matrix.sum(A,1) # maybe there is a problem
  offset = np.array([0])
  D = dia_matrix((D.transpose(),offset),shape=(D.shape[0],D.shape[0]))
  L = D-A
  return L

def k_samples(k,samples,indim):
  # This function selects the top k samples based on the Laplacian operator and removes them from the original samples.
  # It returns the selected samples, their indexes, and the remaining samples.
  L = laplace_operator(samples[:,:indim],10)
  score = abs(L*samples[:,indim])
  indexes = score.argsort()[-int(k):][::-1]
  return samples[indexes], indexes,np.delete( samples,indexes,axis=0)

def createBlendFunction(xValidationSorted,quads,tree,idx=0):
   # This function creates a blending function for the given sorted validation data and quadrants, using the provided tree and index.
   # It returns the blending function and unique values within the blending range.    right= tree.rigth_cut_off_values[idx]
    left = tree.left_cut_off_values[idx]
    
    inbetween = torch.where((xValidationSorted[:,idx] >= left) & (xValidationSorted[:,idx] <= right))[0]
    
    m = 1/(left-right)
   
    b = 1-m*left
  
    function_not_unique = linearfunction(xValidationSorted[inbetween,idx],m,b)
    
    unique = xValidationSorted[:,idx].unique(False)
    inbetween_unique = torch.where((unique >= left) & (unique <= right))[0] # maybe here?
    function = linearfunction(unique[inbetween_unique],m,b)
    blendfunctions = []

    for i in range(0,len(quads)):
    
      if function.device.type == 'cpu':
        blend = torch.ones((xValidationSorted.shape[0]))
        blend[torch.where(xValidationSorted[:,idx]<left)]= 0
        
        blend[inbetween] = function_not_unique
        if (quads[i][:,idx] >= left).all(): 
          blend = 1-blend
      else:
       
        blend_unique = torch.ones((unique.shape[0])).to(device)
        blend_unique[torch.where(unique >= left)]= 0
        
        blend_unique[inbetween_unique] = function        

        if (quads[i][:,idx] >= left).all(): 
          blend_unique = 1-blend_unique
        
        # create a function that in takes the unique blend and transform it into 
        blend = torch.ones((xValidationSorted.shape[0])).to(device)
        
        for i in range(0,unique.shape[0]):
          indexes = torch.where(xValidationSorted[:,idx] == unique[i])[0]
          blend[indexes] = blend_unique[i]
      
      blendfunctions.append(blend.reshape((blend.shape[0],1)))
      
    
    return blendfunctions

# TODO Check if this transformation makes sense
def transform(quad,func,xyVal):
    """
    Applies the given function to the given validation data (xyVal) within the boundaries of the provided quadrant.
    The quadrant's boundaries are determined by finding the minimum and maximum values for each dimension in the quadrant.

    Parameters:
    quad (torch tensor): Tensor containing the data for a quadrant.
    func (torch tensor): Tensor containing a function to be applied to the data.
    xyVal (torch tensor): Tensor containing the validation data.

    Returns:
    functs (torch tensor): Tensor containing the transformed data.
    """

    max = []
    min = []
    for dimension in range(0,xyVal.shape[-1]-1):
        max.append(torch.max(quad[:,dimension]))
        min.append(torch.min(quad[:,dimension]))


    functs = torch.zeros((xyVal.shape[0],1)).to(device)

    if xyVal.shape[-1] ==1: 
      quadinbex = torch.where((xyVal[:,0] <= max[0]) & (xyVal[:,0] >= min[0]))
      functs[quadinbex] = func.float()
    elif xyVal.shape[-1] ==2: 
      quadinbex = torch.where((xyVal[:,0] <= max[0]) & (xyVal[:,0] >= min[0]) & (xyVal[:,1] <= max[1]) & (xyVal[:,1] >= min[1]))
      functs[quadinbex] = func.float()
    elif xyVal.shape[-1] ==3:
      quadinbex = torch.where((xyVal[:,0] <= max[0]) & (xyVal[:,0] >= min[0]) & (xyVal[:,1] <= max[1]) & (xyVal[:,1] >= min[1]) & (xyVal[:,2] <= max[2]) & (xyVal[:,2] >= max[2]))
      functs[quadinbex] = func.float()

    return functs

def sumFunction(blendfunctions,fun,xyVal):
  """
  Calculates the sum of the given blend function and it's corresponding function.
  The sum is calculated element-wise for each value in the given validation data (xyVal).

  Parameters:
  blendfunctions (list of torch tensors): List of tensors containing blend functions.
  fun (list of torch tensors): List of tensors containing functions.
  xyVal (torch tensor): Tensor containing the validation data.

  Returns:
  sumfuncts (torch tensor): Tensor containing the sum of the blend functions and functions.
  """
  if fun[0].device.type == 'cpu':
    sumfuncts = torch.zeros((xyVal.shape[0],1))
  else:
    sumfuncts = torch.zeros((xyVal.shape[0],1)).to(device)
  for i in range(0,len(blendfunctions)):
    sumfuncts+= blendfunctions[i] * fun[i]
  return sumfuncts
