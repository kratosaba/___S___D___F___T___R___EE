import torch
from mlp import MLPflat
from tree_utils import *
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Tree(object):
    """
    Creates a tree structure that contains in each node a MLP class.
     
    """
    def __init__(self):
        self.childs = []
        self.network = None
        self.depth = None
        
        self.rigth_cut_off_values = torch.ones(3)
        self.left_cut_off_values = torch.ones(3)
        self.diffFunction = None

    def createChildren(self,amount):
        for i in range(0,amount):
            self.childs.append(Tree())


    def train(self,k,maxdepth,inTrain,epochs,indim,batchsize,errorTolerance,num_hidden_layers, hidden_features,porcentage=0.5,weightdecay=1e-6,importance_sampling=False):
        """
        Training the tree:
        k: current depthh
        maxdepth: maximum depth
        inTrain: input training set
        epochs
        tree
        batchsize
        errorTolerance
        N: number of hidden layers
        H: 'Hight' of layer
        porcentage: porcentage for importance sampling
        weightdecay: weightdecay
        importance_sampling: If you would like to do the sampling
        """

        phi = MLPflat(indim, 1,num_hidden_layers, hidden_features).to(device) # GPU
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(phi.parameters(), lr = 1e-3,weight_decay=weightdecay)
        
        if importance_sampling:  
            ksamples,indexes,validation = k_samples(inTrain.shape[0]*(porcentage),inTrain.detach().cpu().numpy(),indim) # we find the most different K points

            ksamples,validation = torch.from_numpy(ksamples.astype(np.float32)),torch.from_numpy(validation.astype(np.float32))    
        else:
            ksamples,validation=inTrain,inTrain
        
        
        train_data = TensorDataset(ksamples[:,:indim], ksamples[:,indim])
    
        #batchsize = int(ksamples.shape[0]*0.0024)

        if batchsize < 1:
            batchsize = 2

        train_loader = DataLoader(dataset=train_data, batch_size= batchsize, shuffle=True)
        # Train
        phi.train()

        train_network(phi,train_loader,epochs,criterion,optimizer,k)    
        
        del train_data,train_loader
        
        # Phi getting evaluated 
        with torch.no_grad():
            phi.eval()
            
            yPredictThisLevel = phi(validation[:,:indim].to(device)) # GPU
            
            
            
            outDiff = validation[:,indim].to(device) - yPredictThisLevel[:,0] # GPU
            currAbsError = torch.mean(torch.abs(outDiff))
            domainDiff = validation.clone() # maybe there is a problem here 
            
            domainDiff[:,indim] = outDiff
            self.diffFunction = outDiff.detach().cpu()
        

        quadrants = split_into_quadrants(domainDiff,self) #GPU
        
        del yPredictThisLevel,domainDiff,outDiff
        
        self.network = phi
        self.depth = k
        
        del phi
        
        if (k == maxdepth) or  (currAbsError <= errorTolerance) :
            pass
        elif currAbsError > errorTolerance:
        
            self.createChildren(2**indim)
            
            for i in range(0,len(self.childs)):

                ep = epochs * 2
                b = int(num_hidden_layers)
                h = int(hidden_features)
                weight_decay = 0
                self.childs[i].train(k+1,maxdepth,quadrants[i],ep,indim,batchsize,errorTolerance,b,h,porcentage,weightdecay,importance_sampling)
                
        
        del quadrants
        

    # Here is the problem
    # try to remember how it works now before anything. 
    def evaluate(self,eval_points):
        
        with torch.no_grad():
            self.network.eval()
            funValidationThisLevel = self.network(eval_points)
    
  
        if self.childs:
        
            eval_quadtrants = split_val_quadrants(eval_points,self)
            
            rec_fun = []
            
            # Iterative function.
            for i in range(0,len(eval_quadtrants)):
                rec_fun.append(self.childs[i].evaluate(eval_quadtrants[i]))
            
            tran_func = []
            for i in range(0,len(eval_quadtrants)):
                # Si funciona
                tran_func.append(transform(eval_quadtrants[i],rec_fun[i],eval_points))
            
            
            # Blend functions
            
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


            # Sum results
            funSum = sumFunction(blendfunctions,tran_func,eval_points)
            
            
            
            del rec_fun,blendfunctions
            return funSum + funValidationThisLevel
        else:
            #print(self.depth)
            return funValidationThisLevel
