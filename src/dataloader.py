import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler


class dataset (Dataset):
    def __init__(self,train,height,width,transform=True,intensity=True,seed=42,data_path='../data'):  
        self.height = height
        self.width = width
        self.transform = transform
        self.intensity = intensity
        data_path = os.path.join(data_path, 'train' if train else 'test')
        self.image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        self.image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(self.image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.npy')
        self.rng = np.random.default_rng(seed=seed)
    
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
        mask[image[:,:,4] < 35] = 0 
        mask[image[:,:,4] >= 35] = 1
        mask[:5,:5] = 0
        image[mask==0] = 0
        
        #scale the image up to batch_size and pad remaining
        hscaler = self.height/image.shape[0]
        wscaler = self.width/image.shape[1]
        scaler = np.zeros(2)
        if hscaler > wscaler:
            image = cv2.resize(image,dsize=(0,0),fx=wscaler,fy=wscaler,interpolation=cv2.INTER_LINEAR)
            pad = self.height - image.shape[0]
            top = int(np.floor(pad * 0.5))
            bottom = int(pad-top)
            image = np.pad(image, ((top,bottom),(0,0),(0,0)), 'constant', constant_values=(0,0))
            scaler[1] = wscaler 
            
        elif hscaler < wscaler:
            image = cv2.resize(image,dsize=(0,0),fx=hscaler,fy=hscaler,interpolation=cv2.INTER_LINEAR)
            pad = self.width - image.shape[1]
            left = int(np.floor(pad * 0.5))
            right = int(pad-left)
            image = np.pad(image, ((0,0),(left,right),(0,0)), 'constant', constant_values=(0,0))
            scaler[0] = hscaler
        else:
            #Perfect fit no padding needed
            image = cv2.resize(image,dsize=(0,0),fx=wscaler,fy=hscaler,interpolation=cv2.INTER_LINEAR)
            scaler[0] = hscaler
        
        #image = cv2.resize(image,self.size,interpolation=cv2.INTER_LINEAR)
        if self.transform:
            flips = lambda x: [np.fliplr(x), np.flipud(x), np.flipud(np.fliplr(x)), x]
            image = self.rng.choice(flips(image))
        
        X = transforms.functional.to_tensor(image)
        
        if self.intensity:
            for i in range(X.shape[0]):
                X[i,:,:] = X[i,:,:]/torch.max(X[i,:,:])
        
        scaler = torch.from_numpy(scaler).float()
        
        return X,y,scaler
    
    def get_image_paths(self):
        return self.image_paths
    
    def get_image_classes(self):
        return self.image_classes



def make_dataloaders(height=128,width=64,batch_size=256,transform=True,intensity=True,weighted=True,seed=42):
    """
    Creates a train and test dataloader with a variable batch size and image shape.
    And using a weighted sampler for the training dataloader to have balanced mini-batches when training.
    """
    train_set = dataset(train=True,transform=transform,intensity=intensity,height=height,width=width,seed=seed)
    test_set = dataset(train=False,transform=False,intensity=intensity,height=height,width=width)
    
    if weighted:
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
        sampler = WeightedRandomSampler(weights=weights,num_samples=len(train_set),replacement=False)
        train_loader = DataLoader(train_set, batch_size=batch_size,sampler=sampler, worker_init_fn=np.random.seed(seed),num_workers=4, pin_memory=True)
        
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, worker_init_fn=np.random.seed(seed),num_workers=4, pin_memory=True)
        
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=np.random.seed(seed),num_workers=4, pin_memory=True)
    
    return train_loader,test_loader
