import sys
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from multicalib.utils import train_predictor
from multicalib.models import IncomeDataset, CreditDataset, NNetPredictor

sys.path.append('..')

class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = []

        # Load the datasets
        if (self.args.data == 'income'):
            trainset = IncomeDataset(file='adult_train.npz', root_dir=self.args.path+'data/')
            trainloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
            testset = IncomeDataset(file='adult_test.npz', root_dir=self.args.path+'data/')
            testloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
        elif (self.args.data == 'credit'):
            trainset = CreditDataset(file='credit_card_default_train.xls', root_dir=self.args.path+'data/')
            trainloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
            testset = CreditDataset(file='credit_card_default_test.xls', root_dir=self.args.path+'data/')
            testloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)

        # Train a predictor model
        if (self.args.mode == 'train'):
            model = NNetPredictor(trainset.__dim__())
            train_predictor(model, trainloader, epochs=self.args.epochs)
            torch.save(model, self.args.path+'models/checkpoint_'+self.args.data+'.mdl')
            torch.save(model.state_dict(), self.args.path+'models/checkpoint_'+self.args.data+'.pth')
        else:
            model = torch.load(self.args.path+'models/checkpoint_'+self.args.data+'.mdl')
            model.load_state_dict(torch.load(self.args.path+'models/checkpoint_'+self.args.data+'.pth'))

        self.model = model
        self.trainset = trainset
        self.testset = testset

