import torch
from scipy.sparse import coo_matrix , dia_matrix, lil_matrix
from sklearn.neighbors import BallTree
import numpy as np 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linearfunction(X,m,b):
  return m*X+b
  
def split_into_quadrants(points,tree, idx=0):
    cut_off = points.shape[-1]-1
    # TODO This is a hack I have change this 
    if cut_off == 2: 
      cut_off = 3

    if idx < (cut_off):
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
  L = laplace_operator(samples[:,:indim],10)
  score = abs(L*samples[:,indim])
  indexes = score.argsort()[-int(k):][::-1]
  return samples[indexes], indexes,np.delete( samples,indexes,axis=0)

def createBlendFunction(xValidationSorted,quads,tree,idx=0):
  # I have to make a general way to create a blending function. 
  
    right= tree.rigth_cut_off_values[idx]
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
    
    functs = torch.zeros((xyVal.shape[0],1)).to(device)
    if xyVal.shape[-1] ==1: 
      xmaxtvalue,xminvalue = torch.max(quad[:,0]),torch.min(quad[:,0])    
      quadinbex = torch.where((xyVal[:,0] <= xmaxtvalue) & (xyVal[:,0] >= xminvalue))
      functs[quadinbex] = func.float()
    elif xyVal.shape[-1] ==2: 
      xmaxtvalue,xminvalue = torch.max(quad[:,0]),torch.min(quad[:,0])
      ymaxtvalue,yminvalue = torch.max(quad[:,1]),torch.min(quad[:,1])

      quadinbex = torch.where((xyVal[:,0] <= xmaxtvalue) & (xyVal[:,0] >= xminvalue) & (xyVal[:,1] <= ymaxtvalue) & (xyVal[:,1] >= yminvalue))
      functs[quadinbex] = func.float()
    elif xyVal.shape[-1] ==3:
      
      xmaxtvalue,xminvalue = torch.max(quad[:,0]),torch.min(quad[:,0])
      ymaxtvalue,yminvalue = torch.max(quad[:,1]),torch.min(quad[:,1])
      zmaxtvalue,zminvalue = torch.max(quad[:,2]),torch.min(quad[:,2])

      quadinbex = torch.where((xyVal[:,0] <= xmaxtvalue) & (xyVal[:,0] >= xminvalue) & (xyVal[:,1] <= ymaxtvalue) & (xyVal[:,1] >= yminvalue) & (xyVal[:,2] <= zmaxtvalue) & (xyVal[:,2] >= zminvalue))
      functs[quadinbex] = func.float()

    return functs

def sumFunction(blendfunctions,fun,xyVal):
  if fun[0].device.type == 'cpu':
    sumfuncts = torch.zeros((xyVal.shape[0],1))
  else:
    sumfuncts = torch.zeros((xyVal.shape[0],1)).to(device)
  for i in range(0,len(blendfunctions)):
    sumfuncts+= blendfunctions[i] * fun[i]
  return sumfuncts
