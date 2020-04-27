import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import Sampler
from torch.utils.data import WeightedRandomSampler

class dataset (Dataset):
    def __init__(self,data,target):
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        temp = np.load(self.data[idx])
        
        X = torch.from_numpy(temp).type(torch.DoubleTensor)
        y = torch.tensor(self.target[idx]).type(torch.LongTensor)
        
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

batch_size = 512
weights = []
for label in labels_train:
    if label == 0:
        weights.append(0.2/len(labels_train[labels_train==0]))
    elif label == 1:
        weights.append(0.2/len(labels_train[labels_train==1]))
    elif label == 2:
        weights.append(0.2/len(labels_train[labels_train==2]))
    elif label == 3:
        weights.append(0.2/len(labels_train[labels_train==3]))
    else:
        weights.append(0.2/len(labels_train[labels_train==4]))
weights = torch.FloatTensor(weights)
sampler = WeightedRandomSampler(weights=weights,num_samples=len(train_set),replacement=True)

train_loader = DataLoader(train_set, batch_size=batch_size,sampler=sampler,num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,num_workers=0)