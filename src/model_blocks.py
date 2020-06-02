import torch
import torch.nn as nn
class ResNetBlock(nn.Module):
    def __init__(self, n_features, droprate):
        super(ResNetBlock, self).__init__()
        self.conv_bn = nn.Sequential(nn.Conv2d(in_channels=n_features,out_channels=n_features,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(n_features),nn.Dropout(p=droprate))
        self.activation = nn.ReLU()
    
    def forward(self, x):
        identity = x.clone()
        x = self.conv_bn(x)
        x = self.activation(x)
        x = self.conv_bn(x)
        x = x+identity
        out = self.activation(x)
        return out

class SE_ResNetBlock(nn.Module):
    def __init__(self, n_features, droprate, r):
        super(SE_ResNetBlock, self).__init__()
        
        self.conv_bn = nn.Sequential(nn.Conv2d(in_channels=n_features,out_channels=n_features,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(n_features),nn.Dropout(p=droprate))
        self.activation = nn.ReLU()
        self.globalpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Conv2d(in_channels=n_features,out_channels=n_features//r,kernel_size=1,stride=1,padding=0) #nn.Linear(in_features=n_features,out_features=n_features//r) 
        self.fc2 = nn.Conv2d(in_channels=n_features//r,out_channels=n_features,kernel_size=1,stride=1,padding=0) #nn.Linear(in_features=n_features//r,out_features=n_features)
        self.gate = nn.Sigmoid()
    
    def forward(self, x):
        identity = x.clone()
        out = self.conv_bn(x)
        
        out = self.activation(out)
        out = self.conv_bn(out)
        
        se = self.globalpool(out) 
        se = self.fc(se)
        se = self.activation(se)
        se = self.fc2(se)
        se = self.gate(se)
        
        out = (out*se)+identity
        out = self.activation(out)
        return out
