import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
  
#Define focal loss    
def focal(outputs,targets,alpha=1,gamma=2):
    ce_loss = F.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean() # mean over the batch
    return focal_loss
    
#Define the training as a function.
def train(model, optimizer, scheduler, train_loader, test_loader, device, batch_size='128', num_epochs=1, model_choice='ConvNet', n_features=16, height=256, width=128, droprate=0.5, lr=0.1, num_blocks=3, r='r'):
    train_acc_all = []
    test_acc_all = []
    classes = test_loader.dataset.get_image_classes()
    writer = SummaryWriter(log_dir="../logs/" + 
    datetime.today().strftime('%d-%m-%y:%H%M') + f' {model_choice} lr={lr} droprate={droprate} blocks={num_blocks} features={n_features} height={height}')
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
            loss = F.cross_entropy(output,target) #focal(output,target) #F.nll_loss(torch.log(output), target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()
            
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
            
            #Remove mini-batch from memory
            del data, target, loss
#            print("mini-batch done")
        #Comput the test accuracy
        test_correct = 0
        model.eval()
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        for data, target in test_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
            predicted = output.argmax(1).cpu()
            
            test_correct += (target == predicted).sum().item()
            
            c = (predicted == target).squeeze()
            for i in range(data.shape[0]):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
        scheduler.step()
                
        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))            

        Barley_Acc = 100 * class_correct[0] / class_total[0]
        Broken_Acc = 100 * class_correct[1] / class_total[1]
        Oat_Acc = 100 * class_correct[2] / class_total[2]
        Rye_Acc = 100 * class_correct[3] / class_total[3]
        Wheat_Acc = 100 * class_correct[4] / class_total[4]

        train_acc = train_correct/len(train_loader.dataset)
        test_acc = test_correct/len(test_loader.dataset)
        writer.add_scalars('Train_Test_Accuracies', {'Train_Accuracy':train_acc, 'Test_Accuracy':test_acc}, epoch)
        writer.add_scalars('Class_Accuracies', {'Barley':Barley_Acc, 'Broken':Broken_Acc, 'Oat':Oat_Acc, 'Rye':Rye_Acc, 'Wheat':Wheat_Acc}, epoch)
        
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)
        
        
        print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))
    writer.add_hparams({'Batch_Size':batch_size, 'Epochs':num_epochs, 'Model':model_choice, 'Features':n_features, 'Height':height, 'Width':width, 'Drop':droprate, 'LR':lr, 'Blocks':num_blocks, 'R':r}, {'hparam/Barley':Barley_Acc, 'hparam/Broken':Broken_Acc, 'hparam/Oat_Acc':Oat_Acc, 'hparam/Rye':Rye_Acc, 'hparam/Wheat':Wheat_Acc, 'hparam/Train_Accuracy':train_acc, 'hparam/Test_Accuracy':test_acc})
    return test_acc_all, train_acc_all
