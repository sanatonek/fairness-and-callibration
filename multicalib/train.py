import sys
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import train_predictor
from models import IncomeDataset, CreditDataset, NNetPredictor

sys.path.append('..')

def train(args):
    args = args
    model = []

    # Load the datasets
    if (args.data == 'income'):
        trainset = IncomeDataset(file='adult_train.npz', root_dir=args.path+'data/')
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    elif (args.data == 'credit'):
        trainset = CreditDataset(file='credit_card_default_train.xls', root_dir=args.path+'data/')
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
 
    # Train a predictor model
    model = NNetPredictor(trainset.__dim__())
    train_predictor(model, trainloader, epochs=args.epochs)
    torch.save(model, args.path+'models/checkpoint_'+args.data+'.mdl')
    torch.save(model.state_dict(), args.path+'models/checkpoint_'+args.data+'.pth')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Use multicalibration to report clibrated results of a model')
    parser.add_argument('--batch_size', type=int, default=100, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--data', type=str, default='income', help='Training epochs')
    parser.add_argument('--path', type=str, default='./', help='Training epochs')
    parser.add_argument('--train',  action='store_true', help='Train the predictor model first')

    args = parser.parse_args()
    train(args)
