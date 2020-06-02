import numpy as np
import os
import sys
import torch
import torch.optim as optim
from datetime import datetime
from training import train
from dataloader import make_dataloaders
# import wandb

# wandb.init()

#Define hyperparameters and model
batch_size = int(sys.argv[3]) 
num_epochs = int(sys.argv[2])
model_choice = sys.argv[1] #"SE_ResNet" #"ResNet" "ConvNet"
n_features = int(sys.argv[8])
height = int(sys.argv[5])
width = int(sys.argv[6])
droprate = float(sys.argv[7])
lr = float(sys.argv[4])
num_blocks = 3
r = 16

metric_params = dict(batch_size=batch_size,
    num_epochs=num_epochs,
    model_choice=model_choice,
    n_features=n_features,
    height=height,
    width=width,
    droprate=droprate,
    lr=lr,
    num_blocks=num_blocks,
    r=r
    )
 

for i in range(len(sys.argv)-1):
    print(sys.argv[i])

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader,test_loader = make_dataloaders(height, width, batch_size)

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
    
# wandb.watch(model)
#initialise optimiser
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.25)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
#run the training loop
#train(model, optimizer, train_loader=train_loader, test_loader=test_loader, device=device, num_epochs=num_epochs)
test_acc_all,train_acc_all = train(model, optimizer, scheduler, train_loader=train_loader, test_loader=test_loader, device=device, **metric_params)

#Save model
today = datetime.today()
torch.save(model.state_dict(), '../Models/{model_choice}_{date}'.format(model_choice=model_choice, date=today.strftime("%I%p-%d-%h")))
np.save('../Models/{model_choice}_test_res_{date}'.format(model_choice=model_choice, date=today.strftime("%I%p-%d-%h")),test_acc_all)
np.save('../Models/{model_choice}_train_res_{date}'.format(model_choice=model_choice, date=today.strftime("%I%p-%d-%h")),train_acc_all)
