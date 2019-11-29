import numpy as np
from functools import reduce

from multicalib.utils import expected_accuracy, calibration_score


def calibrate(data, labels, predictions, sensitive_features, alpha, lmbda):
    calibrated_predictions = predictions.copy()
    print('Total number of samples to begin with: ', len(predictions))
    print('AE pre-calibration: ', abs(np.mean(labels)-np.mean(predictions)))
    v_range = np.arange(0,1,1./lmbda)
    for sensitive_feature in sensitive_features:
        # Find the two subset of the sensitive feature
        sensitive_set = [i for i in range(len(data)) if data[i, sensitive_feature] == 1]
        sensitive_set_not = list(set(range(len(data))) - set(sensitive_set))
        # print('Samples in each subgroup: ', len(sensitive_set), len(sensitive_set_not))
        for S in [sensitive_set, sensitive_set_not]:
            # E_s = np.mean(lables[S])
            change = 1
            while change > 0:
                change = 0
                for v in v_range:
                    S_v = [i for i in S if calibrated_predictions[i]<v+(1./lmbda) and calibrated_predictions[i]>=v]
                    # print('Cheking bin %.1f of size s_v: '%v, len(S_v))
                    if len(S_v)< alpha*lmbda*len(S):
                        continue
                    E_predictions = np.mean(calibrated_predictions[S_v])    # V_hat
                    r = oracle(S_v, E_predictions, alpha/4, labels)
                    if r!=100:
                        # print('Update value: ', r)
                        calibrated_predictions[S_v] = calibrated_predictions[S_v] + (r-E_predictions)

                    if (calibrated_predictions[S_v]<0).any() or (calibrated_predictions[S_v]>1).any():
                        calibrated_predictions[S_v] = normalize(calibrated_predictions[S_v])
                    if set(S_v)!=set([i for i in S if calibrated_predictions[i] < v + (1. / lmbda) and calibrated_predictions[i] >= v]):
                        change += 1
            # print('Accuracy for sensitive feature %d: '%sensitive_feature, expected_accuracy(labels[S],  predictions[S], calibrated_predictions[S]))
    return calibrated_predictions


def multicalibrate(data, labels, predictions, sensitive_features, alpha, lmbda):
    # calibrated_predictions = predictions.copy()
    # feature_list = list(range(len(sensitive_features)))
    # ps = lambda s: reduce(lambda P, x: P + [subset | {x} for subset in P], s, [set()])
    # subset_features = ps(feature_list)[1:]
    # calibrated_predictions = calibrate(data, labels, predictions, subset_features, alpha, lmbda)

    calibrated_predictions = predictions.copy()
    print('Total number of samples to begin with: ', len(predictions))
    print('AE pre-multicalibration: ', abs(np.mean(labels)-np.mean(predictions)))
    v_range = np.arange(0,1,1./lmbda)
    multicalibrate_sets = all_subsets(data,sensitive_features)
    print('%d sets to evaluate'%(len(multicalibrate_sets)))
    for S in multicalibrate_sets:
        if len(S) ==0 :
            continue
        change = 1
        while change > 0:
            change = 0
            for v in v_range:
                S_v = [i for i in S if calibrated_predictions[i]<v+(1./lmbda) and calibrated_predictions[i]>=v]
                # print('Cheking bin %.1f of size s_v: '%v, len(S_v))
                if len(S_v)<= alpha*lmbda*len(S):
                    continue
                print(alpha*lmbda*len(S), "%%%%", len(S_v))
                E_predictions = np.mean(calibrated_predictions[S_v])    # V_hat
                r = oracle(S_v, E_predictions, alpha/4, labels)
                if r!=100:
                    # print('Update value: ', r)
                    calibrated_predictions[S_v] = calibrated_predictions[S_v] + (r-E_predictions)

                if (calibrated_predictions[S_v]<0).any() or (calibrated_predictions[S_v]>1).any():
                    calibrated_predictions[S_v] = normalize(calibrated_predictions[S_v])
                if set(S_v)!=set([i for i in S if calibrated_predictions[i] < v + (1. / lmbda) and calibrated_predictions[i] >= v]):
                    change += 1
        print('Multicalibration Accuracy for S :', expected_accuracy(labels[S],  predictions[S], calibrated_predictions[S])) # doesn't give that much information like calibration
    return calibrated_predictions


def oracle(set, v_hat, omega, labels):
    ps = np.mean(labels[set])
    r=0
    if abs(ps-v_hat)<2*omega:
        r =  100
    if abs(ps-v_hat)>4*omega:
        r =  np.random.uniform(0, 1)
    if r!=100:
        r = np.random.uniform(ps-omega, ps+omega)
    return r


def normalize(x):
    if min(x)==max(x):
        return x/min(x)
    else:
        return (x-min(x))/(max(x)-min(x))


def data_matches_features(data, selected_sensitives, all_sensitives): #input data is a single row in real data
    is_sensitive_required = dict()
    for sensitive_feature in all_sensitives:
        is_sensitive_required[sensitive_feature] = 0
    for required_sensitive_feature in selected_sensitives:
        is_sensitive_required[required_sensitive_feature] = 1
    
    for i in range(len(data)):
        if i in is_sensitive_required:
            if data[i] != is_sensitive_required[i]:
                return False
    
    return True


def all_subsets(data, sensitive_features):# all possible subsets of data with all 2^x values of sensitive feature set
    all_subsets =[]
    x = len(sensitive_features)
    for i in range(1 << x):
        current_subset = []
        for j in range(x):
            if((i>>j)&1):
                current_subset.append(j) 
        all_subsets.append(current_subset) 
    target_sets=[]
    for i in range(1,len(all_subsets)):
        required_sensitive_features = all_subsets[i]
        #targes_sets[i] = [i for i in range(len(data)) if data[i, sensitive_feature] == 1]
        data_indices_matching_features = [i for i in range(len(data)) if (data[i, list(required_sensitive_features)] == 1).all()]
        # data_indices_matching_features = [idx for idx in range(len(data)) if
        #     data_matches_features(data[idx, :], required_sensitive_features, sensitive_features)]
        target_sets.append(data_indices_matching_features)
    return target_sets

