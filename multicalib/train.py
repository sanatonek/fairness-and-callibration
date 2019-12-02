import sys
import torch
import argparse
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import train_predictor
from models import IncomeDataset, CreditDataset, RecidDataset, NNetPredictor

sys.path.append('..')


def train(args):
    args = args
    model = []
    random.seed(1234)

    # Load the datasets
    if (args.data == 'income'):
        trainset = IncomeDataset(file='adult_train.npz', root_dir=args.path+'data/')
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    elif (args.data == 'credit'):
        trainset = CreditDataset(file='credit_card_default_train.xls', root_dir=args.path+'data/')
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    elif (args.data == 'recidivism'):
        trainset = RecidDataset(file='propublica_data_for_fairml_train.csv', root_dir=args.path+'data/')
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
 
    # Train a predictor model
    print(trainset.__dim__())
    model = NNetPredictor(trainset.__dim__())
    train_predictor(args, model, trainloader, epochs=args.epochs)
    # torch.save(model, args.path+'models/checkpoint_'+args.data+'.mdl')
    torch.save(model.state_dict(), args.path+'models/checkpoint_'+args.data+'_reg_'+str(args.reg)+'.pth')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Use multicalibration to report clibrated results of a model')
    parser.add_argument('--batch_size', type=int, default=100, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--data', type=str, default='income', help='Training epochs')
    parser.add_argument('--path', type=str, default='./', help='Training epochs')
    parser.add_argument('--reg', type=str, default='None', help='choose regularization scheme [None, eqo]')

    args = parser.parse_args()
    train(args)
