
#--future-- is python 2 to python 3 (keeps print and other functions the same across versions)
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

#Module is an abstraction funtion that has special objects that track what happens in the network, layers have different variables, and checks variables that it needs to track.

'''
Tracks weights at each layer
Use 'device' to set gpu vs cpu
'''

class Net(nn.Module):
    def __init__(self):
        #Super calls the constructor superclass (creates instance of class similar to inheritance; i.e. creates NN instance)
        super(Net, self).__init__()
        #Here we are initializing instances of each type of layer with various dimensions
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #Ignore dropout for now (how to extend is beyond scope and several networks we saw don't use it)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        #Set activation function
        x = F.relu(x)
        x = self.conv2(x)
        #May need to be carfeul about randomness for MaxPool, because this could perturb the maximum element 
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        #Change data structure from 2 dimensions to 1 dimension.
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #Set activation function for output
        output = F.log_softmax(x, dim=1)
        return output

'''
Optimizer resets the gradients and other tracked objects
'''
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()#This is half/all of backprop
        optimizer.step()#This is the adam optimizer which is a linear addition and should be usable with our scheme
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval() #Similar to model.train() sets training operators to false (disables some printing and other tools for training)
    test_loss = 0
    correct = 0
    with torch.no_grad(): #Doesn't affect data
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
                #These should be done in enclave in plaintext
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings (we shouldn't need to modify this)
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    #Ignore cuda for now (allows for gpu processing)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #Makes process nonrandom (useful for debugging)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    #Saves the model as an object to load onto gpu later
    model = Net().to(device)
    #Sets optimizer, but doesn't change data
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    #Sets learning rate, but doesn't change data
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        #Scheduler doesn't affect data
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

#Can ignore (just calls program for execution)
if __name__ == '__main__':
    main()
                                                                                          
