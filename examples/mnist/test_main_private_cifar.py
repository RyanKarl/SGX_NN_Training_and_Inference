import torch
from torch.autograd import Function
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from PyTorchIPC import LinearAlt, ConvAlt, LinearAltLast
from optimizer import SGD, MyLoss
import sys

#Force Determinism
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 1024
num_classes = 10
num_epochs = 1
batch_size = 128
learning_rate = [.1,.05,.01,.005,.001][int(sys.argv[1]) - 1]

# MNIST dataset 
train_dataset = torchvision.datasets.CIFAR10(root='data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #self.enclave = LinearAlt()
        self.conv1 = ConvAlt(3, 128, 3, 1, bias = None)
        self.conv2 = ConvAlt(128, 128, 3, 1, bias = None)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = ConvAlt(128, 256, 3, 1, bias = None)
        self.conv4 = ConvAlt(256, 256, 3, 1, bias = None)
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = ConvAlt(256, 512, 3, 1, bias = None)
        self.conv6 = ConvAlt(512, 512, 3, 1, bias = None)
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = LinearAlt(512*4*4, hidden_size, bias = None) 
        # self.tanh = nn.Tanh()
        # self.fc2 = LinearAlt(hidden_size, hidden_size, bias = None)  
        # # self.tanh = nn.Tanh()
        # self.fc3 = LinearAlt(hidden_size, hidden_size, bias = None)
        # # self.tanh = nn.Tanh()
        # self.fc4 = LinearAlt(hidden_size, hidden_size, bias = None)
        # self.tanh = nn.Tanh()
        self.fc5 = LinearAltLast(hidden_size, num_classes, bias = None)
        self.sm = nn.Softmax()

        self.flat = nn.Flatten()



    def forward(self, x):
        
        #out = self.enclave(out)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool2(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.pool3(out)
        out = self.flat(out)
        out = self.fc1(out)
        # out = self.tanh(out)
        #For testing... will move later
        # #out = self.enclave(out, self.fc1.weight)
        # out = self.fc2(out)
        # # out = self.tanh(out)
        # out = self.fc3(out)
        # # out = self.tanh(out)
        # out = self.fc4(out)
        # out = self.tanh(out)
        out = self.fc5(out)
        out = self.sm(out)
        # print(out)
        # print(out)

        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)


# Loss and optimizer
criterion = MyLoss()
optimizer = SGD(model.parameters(), lr=learning_rate, dampening=0, weight_decay=0, nesterov=False)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 3, 32,32).to(device)
        labels = labels.to(device)
        
        images += 1
        
        #for k in images:
        #    k = torch.add(k, rand_mask)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 3, 32,32).to(device)
        labels = labels.to(device)
        outputs = model(images + 1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
