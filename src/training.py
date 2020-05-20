from model import SE_ResNet
from dataloader import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#Define focal loss    
def focal(outputs,targets,alpha=1,gamma=2):
    ce_loss = F.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean() # mean over the batch
    return focal_loss
    
#Define the training as a function.
def train(model, optimizer, num_epochs=10):
    train_acc_all = []
    test_acc_all = []

    for epoch in range(num_epochs):
        model.train()
        #For each epoch
        train_correct = 0
        for minibatch_no, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = focal(output,target) #F.nll_loss(torch.log(output), target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()
            
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
            
            #Remove mini-batch from memory
            del data, target, loss
#             print("mini-batch done")
        #Comput the test accuracy
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
            predicted = output.argmax(1).cpu()
            test_correct += (target==predicted).sum().item()
        train_acc = train_correct/len(train_set)
        test_acc = test_correct/len(test_set)
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)
        print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))
    return test_acc_all, train_acc_all


#create model and sent to device
model = SE_ResNet(n_in=7,n_features=8,image_size=size).float()
model.to(device)
#initialise optimiser
optimizer = optim.SGD(model.parameters(),lr=1e-3)
#run the training loop
test_acc_all,train_acc_all = train(model,optimizer,num_epochs=400)

#Save model
today = datetime.today()
torch.save(model.state_dict(), '../Models/SEResNet-{date}'.format(date=today.strftime("%I%p-%d-%h")))
np.save('../Models/test_res_{}'.format(today.strftime("%I%p-%d-%h")),test_acc_all)
np.save('../Models/train_res_{}'.format(today.strftime("%I%p-%d-%h")),train_acc_all)