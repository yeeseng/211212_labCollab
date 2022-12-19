import pandas
import pandas as pd
import numpy as np
import statistics
import glob
import tqdm
from datetime import datetime, timedelta
import random
from torch.utils.data import Dataset as BaseDataset
import random

# PyTorch Modules
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import models
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.datasets as dsets


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.GRU = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.linear3 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        #imageLevelOutputs = []
        h0 = torch.zeros(self.num_layers * 2, self.hidden_size)
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()

        out, h_n = self.GRU(x, h0)

        print(h_n.size())
        h_n = h_n.view(1, -1)
        print(h_n.size())
        studyLevelOutputs = F.relu(self.linear3(h_n))
        studyLevelOutputs = self.linear4(studyLevelOutputs)

        return studyLevelOutputs

if __name__ == "__main__":
    INPUT_SIZE = 52
    HIDDEN_SIZE = 32
    NUM_LAYERS = 1
    NUM_CLASSES = 32

    seq = BiGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)

    input = torch.rand((12,52))
    output = seq(input)
    print(output)



