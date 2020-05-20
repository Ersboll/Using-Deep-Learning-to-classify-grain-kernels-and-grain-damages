import numpy as np
import torch
import torch.nn as nn
from model_blocks import SE_ResNetBlock
from model_blocks import ResNetBlock

#Define Squeeze and Excitation ResNet
class SE_ResNet(nn.Module):
    def __init__(self, n_in, n_features,image_size, num_blocks=3,r=8):
        super(SE_ResNet, self).__init__()
        #First conv layers needs to output the desired number of features.
        conv_layers =[nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(2,2), #Reduce image size by half
                      nn.Conv2d(n_features,2*n_features,3,1,1),
                      nn.ReLU()]
        
        for i in range(num_blocks):
            conv_layers.append(SE_ResNetBlock(2*n_features,r))
            
        conv_layers.append(nn.Sequential(nn.MaxPool2d(2,2),
                            nn.Conv2d(2*n_features, 4*n_features, kernel_size=3, stride=1, padding=1),
                            nn.ReLU())) #Reduce image size by half
        
        for i in range(num_blocks):
            conv_layers.append(SE_ResNetBlock(4*n_features,r))
            
        self.blocks = nn.Sequential(*conv_layers)
        
        self.fc = nn.Sequential(nn.Linear(int(image_size[0]/4)*int(image_size[1]/4)*4*n_features, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Linear(512,5))
        
    def forward(self, x):
        x = self.blocks(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
    
#Define Residual network
class ResNet(nn.Module):
    def __init__(self, n_in, n_features,image_size, num_blocks=3):
        super(SE_ResNet, self).__init__()
        #First conv layers needs to output the desired number of features.
        conv_layers =[nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(2,2), #Reduce image size by half
                      nn.Conv2d(n_features,2*n_features,3,1,1),
                      nn.ReLU(),
                     ]
        
        for i in range(num_blocks):
            conv_layers.append(ResNetBlock(2*n_features))
            
        conv_layers.append(nn.Sequential(nn.MaxPool2d(2,2),
                            nn.Conv2d(2*n_features, 4*n_features, kernel_size=3, stride=1, padding=1),
                            nn.ReLU())) #Reduce image size by half
        
        for i in range(num_blocks):
            conv_layers.append(ResNetBlock(4*n_features))
            
        self.blocks = nn.Sequential(*conv_layers)
        
        self.fc = nn.Sequential(nn.Linear(int(image_size[0]/4)*int(image_size[1]/4)*4*n_features, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Linear(512,5))
        
    def forward(self, x):
        x = self.blocks(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
    
#Define convolutional network
class ConvNet(nn.Module):
    def __init__(self, n_in, n_features,image_size):
        super(SE_ResNet, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),
                                   nn.MaxPool2d(2,2), #Reduce image size by half
                                   nn.Batchnorm2d(n_features),
                                   nn.ReLU(),
                                   nn.Conv2d(n_features,2*n_features,3,1,1),
                                   nn.Batchnorm2d(2*n_features),
                                   nn.ReLU(),
                                   nn.Conv2d(2*n_features,2*n_features,3,1,1),
                                   nn.MaxPool2d(2,2), #Reduce image size by half
                                   nn.Batchnorm2d(2*n_features),nn.ReLU(),
                                   nn.ReLU(),
                                   nn.Conv2d(2*n_features,4*n_features,3,1,1),
                                   nn.Batchnorm2d(4*n_features),
                                   nn.ReLU())
        
        self.fc = nn.Sequential(nn.Linear(int(image_size[0]/4)*int(image_size[1]/4)*4*n_features, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Linear(512,5))
        
    def forward(self, x):
        x = self.blocks(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out