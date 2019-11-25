import sys
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
sys.path.append('..')

from train import Trainer
from multicalib.models import IncomeDataset, CreditDataset, NNetPredictor
from multicalib.utils import train_predictor
from multicalib.multicalibration import calibrate,multi_calibrate

def main(args, features=[0,1]):
    # Load the datasets
    trn = Trainer(args)

    x = torch.stack([sample[0] for sample in list(trn.trainset)])[:100]
    y = torch.stack([sample[1] for sample in list(trn.trainset)])[:100]
    print('Data size: ', x.shape)
    predictions = trn.model(x)
    predictions = torch.FloatTensor(predictions.shape).uniform_(0,1)

    # Calibrate output
    multi_calibrate(data=x.numpy(), lables=y.numpy(), predictions=predictions.detach().numpy(), sensitive_features=features, alpha=0.01, lmbda=5)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Use multicalibration to report clibrated results of a model')
    parser.add_argument('--data', type=str, default='income', help='Dataset to use for the experiment')
    parser.add_argument('--features', type=int, default=1, help='List of sensitive attributes')
    parser.add_argument('--batch_size', type=int, default=100, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--mode', type=str, default='calib', help='["train" or "calib"]')
    parser.add_argument('--path', type=str, default='./', help='Root directory path')

    args = parser.parse_args()
    print(args)
    main(args)

