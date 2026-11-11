import torch
import torch.nn as nn
import numpy as np

def data_loading(dataset,batch_size,context_length,device=None):
    n = len(dataset)

    starts = np.random.randint(0,n-context_length,batch_size)

    x = np.stack([dataset[i:i+context_length] for i in starts])
    y = np.stack([dataset[i+1:i+1+context_length] for i in starts])

    x = torch.tensor(x,dtype=torch.long).to(device)
    y = torch.tensor(y,dtype=torch.long).to(device)


            

    return x,y
    