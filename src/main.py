import numpy as np
import os
import sys
import random
import torch
import torch.optim as optim
from datetime import datetime
from training import train
from dataloader import make_dataloaders


#Define hyperparameters and model
r = 16
model_choice = sys.argv[1] #"SE_ResNet" "ResNet" "ConvNet" "ConvNetScale"
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
intensity_type = sys.argv[12] #"imagechannel" "image" "channel"
transform = int(sys.argv[13])
weighted = int(sys.argv[14])
final = int(sys.argv[15])
try:
    seed = int(sys.argv[16])
except:
    seed = np.random.randint(0,2**32-1)

metric_params = dict(batch_size=batch_size,
                     num_epochs=num_epochs,
                     model_choice=model_choice,
                     n_features=n_features,
                     height=height,
                     width=width,
                     droprate=droprate,
                     lr=lr,
                     num_blocks=num_blocks,
                     r=r,
                     weighted=weighted,
                     transform=transform,
                     intensity=intensity,
                     intensity_type=intensity_type)
 
   
print("Model: "+model_choice)
print("Loss function: "+loss_function)
print("Epochs: "+str(num_epochs))
print("Batch size: "+str(batch_size))
print("Learning rate: "+str(lr))
print("Dropout rate: "+str(droprate))
print("Features: "+str(n_features))
print("Number of blocks: "+str(num_blocks))
print("Intensity augmentation: "+str(intensity))
print("Intensity augmentation method: "+intensity_type)
print("Random flip/mirroring: "+str(transform))
print("Weighted sampler: "+str(weighted))
print("Height: "+str(height))
print("Width: "+str(width))
print("Seed: "+str(seed))

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not final:
    train_loader,test_loader = make_dataloaders(height, width, batch_size,transform=transform,intensity=intensity,weighted=weighted,seed=seed,intensity_type=intensity_type,final=final)
else:
    train_loader = make_dataloaders(height, width, batch_size,transform=transform,intensity=intensity,weighted=weighted,seed=seed,intensity_type=intensity_type,final=final)
    test_loader = None

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
    
elif model_choice == "ConvNetScale":
    from model import ConvNetScale
    model = ConvNetScale(n_in=7, n_features=n_features, height=height, width=width, droprate=droprate).float()
    model.to(device)
    print("ConvNetScale initialized")

else:
    sys.exit("The chosen model isn't valid")
    
#initialise optimiser
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.05)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(num_epochs), gamma=0.1)
#run the training loop
#train(model, optimizer, train_loader=train_loader, test_loader=test_loader, device=device, num_epochs=num_epochs)
train(model, optimizer, scheduler, train_loader=train_loader, test_loader=test_loader, device=device, loss_function=loss_function, seed=seed, final=final, **metric_params)
