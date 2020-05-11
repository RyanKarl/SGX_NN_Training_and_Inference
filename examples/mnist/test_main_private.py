import torch
from torch.autograd import Function
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from PyTorchIPC import LinearAlt, ConvAlt, LinearAltLast, my_cross_entropy
from optimizer import SGD, MyLoss

#Force Determinism
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 128
learning_rate = .1

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
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
        # self.conv1 = ConvAlt(1, 32, 3, 1, bias = None)
        # self.conv2 = ConvAlt(32, 64, 3, 1, bias = None)
        # self.conv3 = ConvAlt(64, 128, 3, 1, bias = None)
        self.fc1 = LinearAlt(784, hidden_size, bias = None) 
        # self.tanh = nn.Tanh()
        self.fc2 = LinearAlt(hidden_size, hidden_size, bias = None)  
        # # self.tanh = nn.Tanh()
        self.fc3 = LinearAlt(hidden_size, hidden_size, bias = None)
        # # self.tanh = nn.Tanh()
        self.fc4 = LinearAlt(hidden_size, hidden_size, bias = None)
        # self.tanh = nn.Tanh()
        self.fc5 = LinearAltLast(hidden_size, num_classes, bias = None)
        # self.sm = nn.Softmax()

        self.flat = nn.Flatten()



    def forward(self, x):
        
        #out = self.enclave(out)
        # out = self.conv1(x)
        # out = self.conv2(out)
        # out = self.conv3(out)
        out = self.flat(x)
        out = self.fc1(out)
        # out = self.tanh(out)
        #For testing... will move later
        # #out = self.enclave(out, self.fc1.weight)
        out = self.fc2(out)
        # out = self.tanh(out)
        # out = self.fc3(out)
        # # out = self.tanh(out)
        # out = self.fc4(out)
        # out = self.tanh(out)
        out = self.fc5(out)
        # out = self.sm(out)
        # print(out)
        # print(out)

        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)


# Loss and optimizer
criterion = my_cross_entropy
optimizer = SGD(model.parameters(), lr=learning_rate, dampening=0, weight_decay=0, nesterov=False)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28,28, 1).to(device)
        labels = labels.to(device)
        
        # images += 1
        
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
        images = images.reshape(-1, 28,28, 1).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')