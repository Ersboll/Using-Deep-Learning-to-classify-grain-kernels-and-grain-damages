# import torch
import torch.nn as nn
from torch import cat
from model_blocks import SE_ResNetBlock
from model_blocks import ResNetBlock

#Define Squeeze and Excitation ResNet
class SE_ResNet(nn.Module):
    def __init__(self, n_in, n_features, height, width, droprate, num_blocks=3,r=16):
        super(SE_ResNet, self).__init__()
        #First conv layers needs to output the desired number of features.
        conv_layers =[nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),
                      nn.Dropout(p=droprate),
                      nn.ReLU()]
        
        for i in range(num_blocks):
            conv_layers.append(SE_ResNetBlock(n_features,droprate))
            
        conv_layers.append(nn.Sequential(nn.MaxPool2d(2,2),
                                         nn.Conv2d(n_features, 2*n_features, kernel_size=3, stride=1, padding=1),
                                         nn.Dropout(p=droprate),
                                         nn.ReLU())) #Reduce image size by half
                           
        for i in range(num_blocks):
            conv_layers.append(SE_ResNetBlock(2*n_features,droprate))
            
        conv_layers.append(nn.Sequential(nn.MaxPool2d(2,2),
                                         nn.Conv2d(2*n_features, 4*n_features, kernel_size=3, stride=1, padding=1),
                                         nn.Dropout(p=droprate),
                                         nn.ReLU()))
            
        self.blocks = nn.Sequential(*conv_layers)
        
        self.fc = nn.Sequential(nn.Linear(int(height/4)*int(width/4)*4*n_features, 1024),
                                nn.Dropout(p=droprate),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.Dropout(p=droprate),
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
    def __init__(self, n_in, n_features, height, width, droprate, num_blocks=3):
        super(ResNet, self).__init__()
        #First conv layers needs to output the desired number of features.
        conv_layers =[nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),
                      nn.Dropout(p=droprate),
                      nn.ReLU()]
        
        for i in range(num_blocks):
            conv_layers.append(ResNetBlock(n_features,droprate))
            
        conv_layers.append(nn.Sequential(nn.MaxPool2d(2,2),
                                         nn.Conv2d(n_features, 2*n_features, kernel_size=3, stride=1, padding=1),
                                         nn.Dropout(p=droprate),
                                         nn.ReLU())) #Reduce image size by half
                           
        for i in range(num_blocks):
            conv_layers.append(ResNetBlock(2*n_features,droprate))
            
        conv_layers.append(nn.Sequential(nn.MaxPool2d(2,2),
                                         nn.Conv2d(2*n_features, 4*n_features, kernel_size=3, stride=1, padding=1),
                                         nn.Dropout(p=droprate),
                                         nn.ReLU()))
            
        self.blocks = nn.Sequential(*conv_layers)
        
        self.fc = nn.Sequential(nn.Linear(int(height/4)*int(width/4)*4*n_features, 1024),
                                nn.Dropout(p=droprate),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.Dropout(p=droprate),
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
    def __init__(self, n_in, n_features, height, width, droprate):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),                                                          
                                   nn.BatchNorm2d(n_features),
                                   nn.Dropout(p=droprate),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2,2), #Reduce image size by half  
                                   nn.Conv2d(n_features,2*n_features,3,1,1),
                                   nn.BatchNorm2d(2*n_features),
                                   nn.Dropout(p=droprate),
                                   nn.ReLU(),
                                   nn.Conv2d(2*n_features,2*n_features,3,1,1),                                 
                                   nn.BatchNorm2d(2*n_features),nn.ReLU(),
                                   nn.Dropout(p=droprate),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2,2), #Reduce image size by half
                                   nn.Conv2d(2*n_features,4*n_features,3,1,1),
                                   nn.BatchNorm2d(4*n_features),
                                   nn.Dropout(p=droprate),
                                   nn.ReLU())
        
        self.fc = nn.Sequential(nn.Linear(int(height/4)*int(width/4)*4*n_features, 1024),
                                nn.Dropout(p=droprate),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.Dropout(p=droprate),
                                nn.ReLU(),
                                nn.Linear(512,5))
        
    def forward(self, x):
        x = self.conv1(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
    
#Define convolutional network
class ConvNetScale(nn.Module):
    def __init__(self, n_in, n_features, height, width, droprate):
        super(ConvNetScale, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),                                                          
                                   nn.BatchNorm2d(n_features),
                                   nn.Dropout(p=droprate),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2,2), #Reduce image size by half  
                                   nn.Conv2d(n_features,2*n_features,3,1,1),
                                   nn.BatchNorm2d(2*n_features),
                                   nn.Dropout(p=droprate),
                                   nn.ReLU(),
                                   nn.Conv2d(2*n_features,2*n_features,3,1,1),                                 
                                   nn.BatchNorm2d(2*n_features),nn.ReLU(),
                                   nn.Dropout(p=droprate),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2,2), #Reduce image size by half
                                   nn.Conv2d(2*n_features,4*n_features,3,1,1),
                                   nn.BatchNorm2d(4*n_features),
                                   nn.Dropout(p=droprate),
                                   nn.ReLU())
        
        self.fc = nn.Sequential(nn.Linear(int(height/4)*int(width/4)*4*n_features+2, 1024),
                                nn.Dropout(p=droprate),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.Dropout(p=droprate),
                                nn.ReLU(),
                                nn.Linear(512,5))
        
    def forward(self, x, scaler):
        x = self.conv1(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = cat((x,scaler),1)
        out = self.fc(x)
        return out


