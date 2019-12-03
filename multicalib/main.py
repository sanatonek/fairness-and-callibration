import sys
import torch
import argparse
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
sys.path.append('..')

from multicalib.models import IncomeDataset, CreditDataset, RecidDataset, NNetPredictor
from multicalib.utils import expected_accuracy, calibration_score, EqualizedOddsReg
from multicalib.multicalibration import calibrate, multicalibrate


sensitive_features = {
                      # 1:age_u30, 65: race_black, 66: female
                      'income':[1, 65, 66], 
                      # 2: sex, 5: age
                      'credit': [2, 5], 
                      # 3: Age>45, 4: Age<25, 5: Black, 6: Asian, 7: Hispanic, 10: Female
                      'recidivism': [5, 10]
                     }


def main(args):
    # Load the datasets
    print('Loading %s dataset'%args.data)
    if args.data == 'income':
        testset = IncomeDataset(file='adult_test.npz', root_dir=args.path+'data/')
        testloader = DataLoader(testset, batch_size=100, shuffle=True)
    elif args.data == 'credit':
        testset = CreditDataset(file='credit_card_default_test.xls', root_dir=args.path+'data/')
        testloader = DataLoader(testset, batch_size=100, shuffle=True)
    elif (args.data == 'recidivism'):
        testset = RecidDataset(file='propublica_data_for_fairml_test.csv', root_dir=args.path+'data/')
        testloader = DataLoader(testset, batch_size=100, shuffle=True)    
    
    features = sensitive_features[args.data]

    # Load a predictor model
    print('Loading the predictor model')
    model = NNetPredictor(testset.__dim__())
    model.load_state_dict(torch.load(args.path+'models/checkpoint_'+args.data+'_reg_None.pth'))
    print('Loading the regularized predictor model')
    model_reg = NNetPredictor(testset.__dim__())
    model_reg.load_state_dict(torch.load(args.path+'models/checkpoint_'+args.data+'_reg_eqo.pth'))

    x = torch.stack([sample[0] for sample in list(testset)])
    y = torch.stack([sample[1] for sample in list(testset)])
    predictions = torch.nn.Sigmoid()(model(x))[:,1]
    predictions_reg = torch.nn.Sigmoid()(model_reg(x))[:, 1]


    calibrated_predictions = calibrate(data=x.numpy(), labels=y.numpy(), predictions=predictions.detach().numpy(),
                                       sensitive_features=sensitive_features[args.data], alpha=args.alpha, lmbda=5)
    multicalibrated_predictions = multicalibrate(data=x.numpy(), labels=y.numpy(), predictions=predictions.detach().numpy(),
                                        sensitive_features=features, alpha=args.alpha, lmbda=5)

    # Evaluate performance
    for sensitive_feature in features:
        print('\n=====> Results for feature ', sensitive_feature)
        print('********** Overal Accuracy **********')
        print('Ground truth: %2f\t Regularized: %2f\t Calibration %2f\t Multicalibration %2f\t'
              %(expected_accuracy(y.numpy(), predictions.detach().numpy()) , expected_accuracy(y.numpy(), predictions_reg.detach().numpy()),
              expected_accuracy(y.numpy(), calibrated_predictions) , expected_accuracy(y.numpy(), multicalibrated_predictions)))
        # Find the two subset of the sensitive feature
        sensitive_set = [i for i in range(len(x)) if x.numpy()[i, sensitive_feature] == 1]

        y_s = y.numpy()[sensitive_set]
        prediction_s = predictions.detach().numpy()[sensitive_set]
        predictions_reg_s = predictions_reg.detach().numpy()[sensitive_set]
        calibrated_predictions_s = calibrated_predictions[sensitive_set]
        multicalibrated_predictions_s = multicalibrated_predictions[sensitive_set]

        ## TPR, TNR
        # ground truth
        tpr = np.dot((prediction_s), (y_s))/ np.sum(y_s)
        tnr = np.dot((prediction_s), (1 - y_s))/ np.sum(1 - y_s)
        # reg
        tpr_reg = np.dot((predictions_reg_s), (y_s))/np.sum(y_s)
        tnr_reg = np.dot((predictions_reg_s), (1 - y_s))/ np.sum(1 - y_s)
        # calib
        tpr_reg_calib = np.dot(calibrated_predictions_s, y_s)/ np.sum(y_s)
        tnr_reg_calib = np.dot(calibrated_predictions_s, (1 - y_s))/ np.sum(1 - y_s)
        # multicalib
        tpr_reg_multicalib = np.dot(multicalibrated_predictions_s, y_s)/ np.sum(y_s)
        tnr_reg_multicalib = np.dot(multicalibrated_predictions_s, (1 - y_s))/ np.sum(1 - y_s)
        print('********** TPR **********')
        print('Ground truth: %2f\t Regularized: %2f\t Calibration %2f\t Multicalibration %2f\t'
              %(tpr, tpr_reg, tpr_reg_calib, tpr_reg_multicalib))
        print('********** TNR **********')
        print('Ground truth: %2f\t Regularized: %2f\t Calibration %2f\t Multicalibration %2f\t'
              %(tnr, tnr_reg, tnr_reg_calib, tnr_reg_multicalib))

        ##  Accuracies
        # Ground truth
        accuracy = expected_accuracy(y_s, prediction_s)
        # reg
        accuracy_reg = expected_accuracy(y_s, predictions_reg_s)
        # calib
        accuracy_calib = expected_accuracy(y_s, calibrated_predictions_s)
        # multicalib
        accuracy_multicalib = expected_accuracy(y_s, multicalibrated_predictions_s)
        print('********** Accuracy **********')
        print('Ground truth: %2f\t Regularized: %2f\t Calibration %2f\t Multicalibration %2f\t'
              %(accuracy, accuracy_reg, accuracy_calib, accuracy_multicalib))


        ##  Calibration
        # Ground truth
        calibration = calibration_score(y_s, prediction_s)
        # reg
        calibration_reg = calibration_score(y_s, predictions_reg_s)
        # calib
        calibration_calib = calibration_score(y_s, calibrated_predictions_s)
        # multicalib
        calibration_multicalib = calibration_score(y_s, multicalibrated_predictions_s)
        print('********** Calibration **********')
        print('Ground truth: %2f\t Regularized: %2f\t Calibration %2f\t Multicalibration %2f\t'
              %(calibration, calibration_reg, calibration_calib, calibration_multicalib))


        # true_acc, calibrated_acc = expected_accuracy(y.numpy()[sensitive_set], predictions.detach().numpy()[sensitive_set], calibrated_predictions[sensitive_set])
        # _, multicalibrated_acc = expected_accuracy(y.numpy()[sensitive_set], predictions.detach().numpy()[sensitive_set], multicalibrated_predictions[sensitive_set])
        # _, reg_acc = expected_accuracy(y.numpy()[sensitive_set], predictions.detach().numpy()[sensitive_set], predictions_reg.detach().numpy()[sensitive_set])
        # # Find calibration score
        # true_score, calibrated_score = calibration_score(y.numpy()[sensitive_set], predictions.detach().numpy()[sensitive_set], calibrated_predictions[sensitive_set])
        # _, multicalibrated_score = calibration_score(y.numpy()[sensitive_set], predictions.detach().numpy()[sensitive_set],multicalibrated_predictions[sensitive_set])
        # _, reg_calibrated_score = calibration_score(y.numpy()[sensitive_set], predictions.detach().numpy()[sensitive_set], predictions_reg.detach().numpy()[sensitive_set])
        # # Find equalized odd score
        # # eq = EqualizedOddsReg()
        # true_eq = eq(y, y, x[:, sensitive_feature])
        # calibrated_eq = eq(torch.Tensor(calibrated_predictions), y, x[:, sensitive_feature])
        # multicalibrated_eq = eq(torch.Tensor(multicalibrated_predictions), y, x[:,sensitive_feature])
        # reg_eq = eq(torch.Tensor(predictions_reg), y, x[:, sensitive_feature])
        #
        #
        # print("=====> Results for feature %d:"%sensitive_feature)
        # print("Original labels: \tAccuracy: %.2f \tCalibration score: %.2f \teq score: %.2f" %(true_acc, true_score, true_eq))
        # print("Regularized labels: \tAccuracy: %.2f \tCalibration score: %.2f \teq score: %.2f " % (reg_acc, reg_calibrated_score, calibrated_eq))
        # print("Calibrated labels: \tAccuracy: %.2f \tCalibration score: %.2f \teq score: %.2f" % (calibrated_acc, calibrated_score, calibrated_eq))
        # print("Multicalibrated labels: \tAccuracy: %.2f \tCalibration score: %.2f \teq score: %.2f" % (multicalibrated_acc, multicalibrated_score, multicalibrated_eq))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Use multicalibration to report clibrated results of a model')
    parser.add_argument('--data', type=str, default='income', help='Dataset to use for the experiment')
    parser.add_argument('--path', type=str, default='./', help='Root directory path')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter for calibration')

    args = parser.parse_args()
    main(args)

