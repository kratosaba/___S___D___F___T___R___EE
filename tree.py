import torch
from mlp import MLPflat
from tree_utils import *
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Tree(object):
    """
    The Tree class represents a tree structure with multiple levels, where each level has a neural network associated with it. It has the following methods:
    """
    def __init__(self):
        """
        This is the constructor for the Tree class. It initializes the following instance variables:

        childs: a list of Tree objects representing the children of the current node
        network: the neural network associated with the current level of the tree
        depth: the depth of the current level in the tree
        right_cut_off_values: the cut-off values for the right child of the current node
        left_cut_off_values: the cut-off values for the left child of the current node
        diffFunction: the difference between the predicted output of the network and the true output
        """
        self.childs = []
        self.network = None
        self.depth = None
        
        self.rigth_cut_off_values = torch.ones(3) 
        self.left_cut_off_values = torch.ones(3)
        self.diffFunction = None

    def createChildren(self,amount):
        """
        This method creates amount children for the current node and appends them to the childs list.
        """
        for i in range(0,amount):
            self.childs.append(Tree())


    def train(self,k,max_depth,train_samples,epochs,in_dimesion,batchsize,errorTolerance,num_hidden_layers, hidden_features,percentage_used_trained_importance=0.5,weightdecay=1e-6,importance_sampling=False):
        """
        This method trains the neural network associated with the current level of the tree. The training process involves the following steps:

        1. Create a neural network with the specified number of input dimensions, hidden layers, and hidden features.
        2. Define the loss function and the optimizer to be used during training.
        3. If importance sampling is enabled, select a subset of the training samples and set them aside for validation. Otherwise, use all the samples for both training and validation.
        4. Load the training data into a DataLoader and train the network using the specified number of epochs.
        5. Evaluate the trained network on the validation data and compute the difference between the predicted and true outputs.
        6. Split the validation data into quadrants and store the resulting quadrants in the quadrants variable.
        7. If the current depth is less than the maximum depth or the absolute error of the network is greater than the error tolerance, create children for the current node and train them using the quadrants data.
        
        """

        phi = MLPflat(in_dimesion, 1,num_hidden_layers, hidden_features).to(device) # GPU
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(phi.parameters(), lr = 1e-3,weight_decay=weightdecay)
        
        if importance_sampling:  
            ksamples,indexes,validation = k_samples(train_samples.shape[0]*(percentage_used_trained_importance),train_samples.detach().cpu().numpy(),in_dimesion) # we find the most different K points

            ksamples,validation = torch.from_numpy(ksamples.astype(np.float32)),torch.from_numpy(validation.astype(np.float32))    
        else:
            ksamples,validation=train_samples,train_samples
        
        train_data = TensorDataset(ksamples[:,:in_dimesion], ksamples[:,in_dimesion])
    

        if batchsize < 1:
            batchsize = 2

        train_loader = DataLoader(dataset=train_data, batch_size= batchsize, shuffle=True)
        
        phi.train()

        train_network(phi,train_loader,epochs,criterion,optimizer,k)    
        
        del train_data,train_loader
        
         
        with torch.no_grad():
            phi.eval()
            
            yPredictThisLevel = phi(validation[:,:in_dimesion].to(device)) # GPU
            
            
            
            outDiff = validation[:,in_dimesion].to(device) - yPredictThisLevel[:,0] # GPU
            currAbsError = torch.mean(torch.abs(outDiff))
            domainDiff = validation.clone() 
            
            domainDiff[:,in_dimesion] = outDiff
            self.diffFunction = outDiff.detach().cpu()
        

        quadrants = split_into_quadrants(domainDiff,self) #GPU
        
        del yPredictThisLevel,domainDiff,outDiff
        
        self.network = phi
        self.depth = k
        
        del phi
        
        if (k == max_depth) or  (currAbsError <= errorTolerance) :
            pass
        elif currAbsError > errorTolerance:
        
            self.createChildren(2**in_dimesion)
            
            for i in range(0,self.childs): 
                ratio =int(quadrants[i].shape[0]/ksamples.shape[0])
                epochs = int(epochs * ratio) 
                num_hidden_layers = int(num_hidden_layers * ratio)
                hidden_features = int(hidden_features * ratio)
                self.childs[i].train(k+1,max_depth,quadrants[i],epochs,in_dimesion,batchsize,errorTolerance,num_hidden_layers,hidden_features,percentage_used_trained_importance,weightdecay,importance_sampling)
                
        
        del quadrants
        

     
    def evaluate(self,eval_points):
        
        with torch.no_grad():
            self.network.eval()
            funValidationThisLevel = self.network(eval_points)
    
  
        if self.childs:
        
            eval_quadtrants = split_val_quadrants(eval_points,self)
            
            rec_fun = []
            
            
            for i in range(0,len(eval_quadtrants)):
                rec_fun.append(self.childs[i].evaluate(eval_quadtrants[i]))
            
            tran_func = []
            for i in range(0,len(eval_quadtrants)):
                
                tran_func.append(transform(eval_quadtrants[i],rec_fun[i],eval_points))
            
            
           
            
            if eval_points.shape[-1] ==1: 
                
                xblendfunctions = createBlendFunction(eval_points,eval_quadtrants,self) 
                yblendfunctions = [1,1] 
                zblendfunctions = [1,1]

            elif eval_points.shape[-1] ==2: 
                
                xblendfunctions = createBlendFunction(eval_points,eval_quadtrants,self) 
                yblendfunctions = createBlendFunction(eval_points,eval_quadtrants,self,1) 
                zblendfunctions = [1,1,1,1]

            elif eval_points.shape[-1] ==3: 
                
                xblendfunctions = createBlendFunction(eval_points,eval_quadtrants,self) 
                yblendfunctions = createBlendFunction(eval_points,eval_quadtrants,self,1) 
                zblendfunctions = createBlendFunction(eval_points,eval_quadtrants,self,2)
                
            blendfunctions = []
            
            for i in range(0,len(xblendfunctions)): blendfunctions.append(xblendfunctions[i]* yblendfunctions[i]* zblendfunctions[i])


            
            funSum = sumFunction(blendfunctions,tran_func,eval_points)
            
            
            
            del rec_fun,blendfunctions
            return funSum + funValidationThisLevel
        else:
            
            return funValidationThisLevel
