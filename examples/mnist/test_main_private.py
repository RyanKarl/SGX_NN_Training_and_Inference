import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np

#Force Determinism
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

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
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #print(input_size, hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size) 
        #print(self.fc1)
        self.tanh = nn.Tanh()
        #print(self.relu)
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        #print(self.fc2)    

    def forward(self, x):
        #print("x: ")
        #print(x) #prints tensor object (matrix of 0s)
        #print("Bias")
        #print(self.fc1.bias)
        #time.sleep(5)
        out = self.fc1(x)
        #print("out")
        #print(out)
        tmp_mask = self.fc1(rand_mask)
        out = torch.sub(out, tmp_mask)
        out = torch.sub(out, self.fc1.bias)
        #print("out: ")
        #print(out)
        #print("Stop")
        #time.sleep(50)
        out = self.tanh(out)
        out = torch.add(out, rand_mask2)
        #print(out)
        out = self.fc2(out)
        tmp_mask = self.fc2(rand_mask2)
        out = torch.sub(out, tmp_mask)
        out = torch.sub(out, self.fc2.bias)
        #print(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        rand_mask = torch.ones(28 * 28)
        rand_mask2 = torch.ones(500)
        #print("Random Mask: ")
        #print(rand_mask)
        #time.sleep(2)
        for k in images:
            #time.sleep(2)
            #print(k)
            k = torch.add(k, rand_mask)
            #time.sleep(2)
            #print(k)
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
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
