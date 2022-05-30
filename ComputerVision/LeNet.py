import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Approximating subsampling described in the paper with a maxpool 2d layer and a relu activation
        self.C1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, padding_mode='zeros')
        self.C2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fcl1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fcl2 = nn.Linear(in_features=120, out_features=84)
        self.fcl3 = nn.Linear(in_features=84, out_features=10)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.C1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.C1(x)), (2, 2))
        x = F.relu(self.fcl1(x.view(-1,)))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)
        x = F.softmax(x)
        return x

    def loss(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true)
