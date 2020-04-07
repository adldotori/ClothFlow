import torch.nn.functional as F
import torch
import torch.nn as nn

class renderLoss(nn.Module):
    def __init__(self, layids = None):
        super(renderLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.L1 = nn.L1Loss()
        self.cross = nn.CrossEntropyLoss()
        # self.nllcrit = nn.NLLLoss2d(size_average=True)

    def forward(self, x, y):
        # x = x.type(torch.long)
        # y = y.type(torch.long)
        return self.loss(x, y) 
