import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import Sampler
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as transforms
import cv2

class dataset (Dataset):
    def __init__(self,train,size,data_path='../data'):  
        self.size = size
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.npy')
    
    def __len__(self):
        return len(self.image_paths) #len(self.data)
    
    def __getitem__(self,idx):        
        image_path = self.image_paths[idx]
        
        image = np.load(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]

        image = image[:,:,:7]
        #create a simple mask, and make everything else 0
        mask = image[:,:,4].copy()
        #fix divide by zero
#         mask[image[:,:,4]/image[:,:,1] < 1] = 0 
#         mask[image[:,:,4]/image[:,:,1] >= 1] = 1
        
        mask[image[:,:,4] < 35] = 0 
        mask[image[:,:,4] >= 35] = 1
        image[mask==0] = 0
        
        image = cv2.resize(image,self.size,interpolation=cv2.INTER_LINEAR)

        X = transforms.functional.to_tensor(image)
        return X,y
    
    def get_image_paths(self):
        return self.image_paths

size=(64,128)
train_set = dataset(train=True,size=size)
test_set = dataset(train=False,size=size)

batch_size = 512
weights = []

train_paths = train_set.get_image_paths()
oat_length = len(os.listdir('../data/train/Oat'))
wheat_length = len(os.listdir('../data/train/Wheat'))
rye_length = len(os.listdir('../data/train/Rye'))
broken_length = len(os.listdir('../data/train/Broken'))
barley_length = len(os.listdir('../data/train/Barley'))

for file in train_paths:
    label = os.path.split(os.path.split(file)[0])[1]
    if label == 'Oat':
        weights.append(0.2/oat_length)
    elif label == "Wheat":
        weights.append(0.2/wheat_length)
    elif label == "Rye":
        weights.append(0.2/rye_length)
    elif label == "Broken":
        weights.append(0.2/broken_length)
    else:
        weights.append(0.2/barley_length)
weights = torch.FloatTensor(weights)
sampler = WeightedRandomSampler(weights=weights,num_samples=len(train_set),replacement=True)

train_loader = DataLoader(train_set, batch_size=batch_size,sampler=sampler,num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,num_workers=0)