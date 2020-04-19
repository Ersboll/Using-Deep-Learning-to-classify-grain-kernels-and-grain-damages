import numpy as np
import torch
import torch.nn as nn
class ResNetBlock(nn.Module):
    def __init__(self, n_features):
        super(ResNetBlock, self).__init__()
        
        self.w1 = nn.Conv2d(in_channels=n_features,out_channels=n_features,kernel_size=3,stride=1,padding=1)
        self.w2 = nn.Conv2d(in_channels=n_features,out_channels=n_features,kernel_size=3,stride=1,padding=1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        identity = x.clone()
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        x = x+identity
        out = self.activation(x)
        return out

class SE_ResNetBlock(nn.Module):
    def __init__(self, n_features,r):
        super(SE_ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=n_features,out_channels=n_features,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_features,out_channels=n_features,kernel_size=3,stride=1,padding=1)
        self.activation = nn.ReLU()
        self.globalpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Conv2d(in_channels=n_features,out_channels=n_features//r,kernel_size=1,stride=1,padding=0) #nn.Linear(in_features=n_features,out_features=n_features//r) 
        self.fc2 = nn.Conv2d(in_channels=n_features//r,out_channels=n_features,kernel_size=1,stride=1,padding=0) #nn.Linear(in_features=n_features//r,out_features=n_features)
        self.gate = nn.Sigmoid()
    
    def forward(self, x):
        identity = x.clone()
        out = self.conv1(x)
        
        out = self.activation(out)
        out = self.conv2(out)
        
        se = self.globalpool(out) #.unsqueeze(-1).unsqueeze(-1) add if using nn.linear
        se = self.fc(se)
        se = self.activation(se)
        se = self.fc2(se)
        se = self.gate(se)
        
        out = out.mul(se.unsqueeze(-1).unsqueeze(-1))+identity
        out = self.activation(out)
        return out
