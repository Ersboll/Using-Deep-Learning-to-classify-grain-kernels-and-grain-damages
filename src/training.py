from model import SE_ResNetBlock
from model import ResNetBlock
import dataloader
import torch.optim as optim
import torch.nn.functional as F

if torch.cuda.is_available():
    print("The code will run on GPU. This is important so things run faster.")
else:
    print("The code will run on CPU. You should probably not do this.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Define network
class SE_ResNet(nn.Module):
    def __init__(self, n_in, n_features, num_blocks=2,r=8):
        super(SE_ResNet, self).__init__()
        #First conv layers needs to output the desired number of features.
        conv_layers =[nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(n_features,n_features,3,1,1),
                      nn.ReLU(),
                      nn.MaxPool2d(2,2), #160x50
                      nn.Conv2d(n_features,2*n_features,3,1,1),
                      nn.ReLU()]
        
        for i in range(num_blocks):
            conv_layers.append(SE_ResNetBlock(2*n_features,r))
            
        conv_layers.append(nn.Sequential(nn.MaxPool2d(2,2),
                            nn.Conv2d(2*n_features, 4*n_features, kernel_size=3, stride=1, padding=1),
                            nn.ReLU())) #80x25
        
        for i in range(num_blocks):
            conv_layers.append(SE_ResNetBlock(4*n_features,r))
            
        conv_layers.append(nn.Sequential(nn.MaxPool2d(2,2),
                            nn.Conv2d(4*n_features, 8*n_features, kernel_size=3, stride=1, padding=1),
                            nn.ReLU())) #40x13 eller #40x12
        for i in range(num_blocks):
            conv_layers.append(SE_ResNetBlock(8*n_features,r))
        
        self.blocks = nn.Sequential(*conv_layers)
        
        self.fc = nn.Sequential(nn.Linear(40*12*8*n_features, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512,5),
                                nn.Softmax(dim=1))
        
    def forward(self, x):
        x = self.blocks(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
    
#Define the training as a function.
def train(model, optimizer, num_epochs=10):
    train_acc_all = []
    test_acc_all = []

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        #For each epoch
        train_correct = 0
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = F.nll_loss(torch.log(output), target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()
            
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
            
            #Remove mini-batch from memory
            torch.cuda.empty_cache()
            del data, target, loss
        #Comput the test accuracy
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
            predicted = output.argmax(1).cpu()
            test_correct += (target==predicted).sum().item()
        train_acc = train_correct/len(trainset)
        test_acc = test_correct/len(testset)
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)
        print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))
    return test_acc_all, train_acc_all


#create model and sent to device
model = SE_ResNet(n_in=7,n_features=8).double()
model.to(device)
#initialise optimiser
optimizer = optim.SGD(model.parameters(),lr=1e-3)
#run the training loop
train(model,optimizer,num_epochs=1)