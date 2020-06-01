import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
    
#Define focal loss    
def focal(outputs,targets,alpha=1,gamma=2):
    ce_loss = F.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean() # mean over the batch
    return focal_loss
    
#Define the training as a function.
def train(model, optimizer, train_loader, test_loader, device, num_epochs=10):
    train_acc_all = []
    test_acc_all = []
    classes = test_loader.dataset.get_image_classes()
    
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
                
        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
#             class_acc["Accuracy of %5s" % (classes[i])] = 100 * class_correct[i] / class_total[i]
            
#         wandb.log(class_acc)
        
        train_acc = train_correct/len(train_loader.dataset)
        test_acc = test_correct/len(test_loader.dataset)
        
        
#         overall_acc["Accuracy of train"] = train_acc
#         overall_acc["Accuracy of test"] = test_acc
#         wandb.log(overall_acc)
        
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)
        
        
        print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))
    return test_acc_all, train_acc_all
