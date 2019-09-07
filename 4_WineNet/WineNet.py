from torch.nn import Module, Linear, Sigmoid, Softmax
import torch

class WineNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_classes):
        super(WineNet, self).__init__()
        self.fc1 = Linear(n_input, n_hidden)
        self.act1 = Sigmoid()
        self.fc2 = Linear(n_hidden, 50)
        self.act2 = Sigmoid()
        self.fc3 = Linear(50, n_classes)
        self.softmax = Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
    
    def inference(self, x):
        x = self.forward(x)
        x = self.softmax(x)
        return x