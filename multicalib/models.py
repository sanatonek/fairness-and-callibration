import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class IncomeDataset(Dataset):
    """Income dataset."""

    def __init__(self, file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.all_data = np.load(self.root_dir +file)
        self.x = self.all_data['x']
        self.y = self.all_data['y']
        self.a = self.all_data['a']
        self.transform = transform

        # Complete all the dataset specific processing here
        print('Income dataset (x) dims: {}'.format(self.x.shape))
        print('Income dataset (y) dims: {}'.format(self.y.shape))
        print('Income dataset (a) dims: {}'.format(self.a.shape))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #    idx = idx.tolist()

        sample_x, sample_y, sample_a = np.array(self.x[idx]), np.array(self.y[idx]), np.array(self.a[idx])
        sample_x, sample_y, sample_a = torch.tensor(sample_x, dtype=torch.float32), torch.tensor(sample_y, dtype=torch.long), torch.tensor \
            (sample_a, dtype=torch.float32)

        # print('sample_x.shape: {}'.format(sample_x.shape))
        # print('sample_y.shape: {}'.format(sample_y.shape))
        # print('sample_a.shape: {}'.format(sample_a.shape))

        return sample_x, sample_y, sample_a


class NNetPredictor(torch.nn.Module):
    def __init__(self):
        """
        Explicit layer definition
        """
        super(NNetPredictor, self).__init__()
        self.fc1 = nn.Linear(113, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 2)

    def forward(self, x):
        """
        Explicit model definition
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x