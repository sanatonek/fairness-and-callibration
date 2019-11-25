import sys
import torch
import argparse
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
sys.path.append('..')

from multicalib.models import IncomeDataset, CreditDataset, NNetPredictor
from multicalib.utils import train_predictor
from multicalib.multicalibration import calibrate


# 1:age_u30, 65: race_black, 66: female
sensitive_features = {'income':[1, 65, 66]}


def main(args):
    # Load the datasets
    print('Loading %s dataset'%args.data)
    if args.data == 'income':
        testset = IncomeDataset(file='adult_test.npz', root_dir=args.path+'data/')
        testloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    elif args.data == 'credit':
        testset = CreditDataset(file='credit_card_default_test.xls', root_dir=args.path+'data/')
        testloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    features = sensitive_features[args.data]

    # Load a predictor model
    print('Loading the predictor model')
    model = torch.load(args.path+'models/checkpoint_'+args.data+'.mdl')
    model.load_state_dict(torch.load(args.path+'models/checkpoint_'+args.data+'.pth'))

    x = torch.stack([sample[0] for sample in list(testset)])
    y = torch.stack([sample[1] for sample in list(testset)])
    predictions = torch.nn.Sigmoid()(model(x))[:,1]

    # Calibrate output
    calibrate(data=x.numpy(), labels=y.numpy(), predictions=predictions.detach().numpy(), sensitive_features=features, alpha=0.1, lmbda=5)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Use multicalibration to report clibrated results of a model')
    parser.add_argument('--data', type=str, default='income', help='Dataset to use for the experiment')
    parser.add_argument('--path', type=str, default='./', help='Root directory path')

    args = parser.parse_args()
    main(args)

