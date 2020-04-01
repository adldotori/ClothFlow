import torch.nn.functional as F
import torch
import torch.nn as nn

class renderLoss(nn.Module):
    def __init__(self, layids = None):
        super(renderLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.L1 = nn.L1Loss()
        self.cross = nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.cross(x, y) 
