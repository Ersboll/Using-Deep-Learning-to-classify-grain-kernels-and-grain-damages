import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import make_dataloaders
import time
import sys

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Define hyperparameters and model
model_choice = sys.argv[1] #"SE_ResNet" "ResNet" "ConvNet"
loss_function = sys.argv[2] #"crossentropy" "focal" 
num_epochs = int(sys.argv[3])
batch_size = int(sys.argv[4]) 
lr = float(sys.argv[5])
width = int(sys.argv[6])
height = int(sys.argv[7])
droprate = float(sys.argv[8])
n_features = int(sys.argv[9])
num_blocks = int(sys.argv[10])
intensity = int(sys.argv[11])
transform = int(sys.argv[12])
weighted = int(sys.argv[13])
r = 16

for i in range(len(sys.argv)):
    print(sys.argv[i])
    
train_loader,test_loader = make_dataloaders(height, width, batch_size,transform=False,intensity=intensity,weighted=weighted)

#initialize model and sent to device
if model_choice == "SE_ResNet":
    from model import SE_ResNet
    model = SE_ResNet(n_in=7, n_features=n_features, height=height, width=width, droprate=droprate, num_blocks=num_blocks, r=r).float()
    model.to(device)
    print("SE_ResNet initialized")
    
elif model_choice == "ResNet":
    from model import ResNet
    model = ResNet(n_in=7, n_features=n_features, height=height, width=width, droprate=droprate, num_blocks=num_blocks).float()
    model.to(device)
    print("ResNet initialized")
    
elif model_choice == "ConvNet":
    from model import ConvNet
    model = ConvNet(n_in=7, n_features=n_features, height=height, width=width, droprate=droprate).float()
    model.to(device)
    print("ConvNet initialized")

else:
    sys.exit("The chosen model isn't valid")
    
model_list = os.listdir("../Models")
model.load_state_dict(torch.load("../Models/{}".format(model_list[0]),map_location=device))
model.eval()

train_loader.sampler.num_samples = 10000

start = time.time()
for data, target,scaler in train_loader:
    data, scaler = data.to(device), scaler.to(device)
    with torch.no_grad():
        output = model(data, scaler)
    predicted = output.argmax(1).cpu()
end = time.time()
diff = (end-start)
print(diff)