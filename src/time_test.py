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
from model import ConvNet
from model import ResNet
import time
import sys

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Define hyperparameters and model
# model_choice = "ConvNet" #"SE_ResNet" "ResNet" "ConvNet"
model_choice = sys.argv[1]
batch_size = int(sys.argv[2])
width = 256
height = 128
droprate = 0
intensity = 1
intensity_type = "channel"

print(batch_size)

class dataset (Dataset):
    def __init__(self,train,height,width,transform=True,intensity=True,seed=42,intensity_type="image",data_path='../data'):  
        self.height = height
        self.width = width
        self.transform = transform
        self.intensity = intensity
        self.intensity_type = intensity_type
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
        mask = np.zeros((image.shape[0],image.shape[1]))
        temp_blue = image[:,:,1].copy()
        temp_blue[temp_blue==0] = 1
        mask[image[:,:,4]/temp_blue >= 1] = 1
        mask[image[:,:,4] >= 40] = 1
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
            if self.intensity_type == "imagechannel":
                for i in range(X.shape[0]):
                    X[i,:,:] = X[i,:,:]/torch.max(X[i,:,:])
            elif self.intensity_type == "image":
                X = X/torch.max(X)
                
            elif self.intensity_type == "channel":
                X[1,:,:] = X[1,:,:]/0.94901961
                X[2,:,:] = X[2,:,:]/0.96078432
                X[3,:,:] = X[3,:,:]/0.99215686
                X[4,:,:] = X[4,:,:]/0.99607843
                X[6,:,:] = X[6,:,:]/0.59215689
 
            else:
                sys.exit("The chosen intensity augmentation method isn't valid")
        
        scaler = torch.from_numpy(scaler).float()
        
        return X,y,scaler
    
    def get_image_paths(self):
        return self.image_paths
    
    def get_image_classes(self):
        return self.image_classes

train_set = dataset(train=True,transform=False,intensity=intensity,height=height,width=width,intensity_type=intensity_type)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

if model_choice == "ResNet":
    model = ResNet(n_in=7, n_features=16, height=128, width=256, droprate=0, num_blocks=2).float()
    model.to(device)
    print("ResNet initialized")
    model.load_state_dict(torch.load("../Models/ResNet_final",map_location=device))
    model.eval()
    
elif model_choice == "ConvNet":
    model = ConvNet(n_in=7, n_features=64, height=128, width=256, droprate=0).float()
    model.to(device)
    print("ConvNet initialized") 
    model.load_state_dict(torch.load("../Models/ConvNet_final",map_location=device))
    model.eval()

print("Model loaded")

train_loader.sampler.num_samples = 10000

start = time.time()
for data, target,scaler in train_loader:
    data = data.to(device)
    with torch.no_grad():
        output = model(data)
    predicted = output.argmax(1).cpu()
end = time.time()
diff = (end-start)
print(diff)