import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

class dataset (Dataset):
    def __init__(self,data,target):
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        temp = np.zeros((len(idx),7,320,100))
        to_be_loaded = self.data[idx]
        for item,i in zip(to_be_loaded,range(len(idx))):
            temp[i,:,:,:] = np.load(item)
        
        X = temp
        y = self.target[idx]
        
        return X,y

data=np.load("data_preprocessed_path.npy")
labels=np.load("labels.npy")

torch.manual_seed(0)
split = random_split(data,(50000,len(data)-50000))

data_train = data[split[0].indices]
data_test = data[split[1].indices]
labels_train = labels[split[0].indices]
labels_test = labels[split[1].indices]

train_set = dataset(data_train,labels_train)
test_set = dataset(data_test,labels_test)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True,num_workers=0)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False,num_workers=0)
