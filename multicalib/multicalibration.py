import numpy as np
from sklearn import preprocessing


def calibrate(data, lables, predictions, sensitive_features, alpha, lmbda):
    predictions = predictions[:,0] # Why!
    lables = lables[:, 0]
    calibrated_predictions = predictions.copy()
    print('Total number of samples to begin with: ', len(predictions))
    print('AE pre-calibration: ', abs(np.mean(lables)-np.mean(predictions)))
    v_range = np.arange(0,1,1./lmbda)
    change = 1
    while change>0:
        print('Only %d sets changed'%change)
        change=0
        for sensitive_feature in sensitive_features:
            # Find the two subset of the sensitive feature
            sensitive_set = [i for i in range(len(data)) if data[i, sensitive_feature] == 1]
            sensitive_set_not = list(set(range(len(data))) - set(sensitive_set))
            print('Samples in each subgroup: ', len(sensitive_set), len(sensitive_set_not))
            for S in [sensitive_set, sensitive_set_not]:
                for v in v_range:
                    S_v = [i for i in S if calibrated_predictions[i]<v+(1./lmbda) and calibrated_predictions[i]>=v]
                    print('Cheking bin %.1f ...... size of s_v: '%v, len(S_v))
                    if len(S_v)==0:
                        continue
                    E_labels = np.mean(lables[S_v])
                    # oracle
                    E_predictions = np.mean(calibrated_predictions[S_v])
                    if abs(E_labels-E_predictions)<alpha/2.:
                        continue
                    update = np.random.uniform(E_labels-alpha/4., E_labels+alpha/4.)
                    print('Update value: ', update)
                    calibrated_predictions[S_v] = calibrated_predictions[S_v] + (update-E_predictions)
                    if (calibrated_predictions[S_v]<0).any() or (calibrated_predictions[S_v]>1).any():
                        calibrated_predictions[S_v] = normalize(calibrated_predictions[S_v])
                    if set(S_v)!=set([i for i in S if calibrated_predictions[i] < v + (1. / lmbda) and calibrated_predictions[i] >= v]):
                        change += 1
    print(calibrated_predictions)
    print(lables)
    print(predictions)
    print('AE post-calibration: ', abs(np.mean(lables)-np.mean(calibrated_predictions)))


def multi_calibrate(data, lables, predictions, sensitive_features, alpha, lmbda):
    predictions = predictions[:,0] # Why!
    lables = lables[:, 0]
    calibrated_predictions = predictions.copy()
    #print('Total number of samples to begin with: ', len(predictions))
    target_sets = all_subsets(data,sensitive_features)
    print('AE pre-calibration: ', abs(np.mean(lables)-np.mean(predictions)))
    v_range = np.arange(0,1,1./lmbda)
    change = 1
    while change>0:
        print('Only %d sets changed')
        change=0 
        #sensitive_set = [i for i in range(len(data)) if data[i, sensitive_feature] == 1]
        #sensitive_set_not = list(set(range(len(data))) - set(sensitive_set))
        #print('Samples in each subgroup: ', len(sensitive_set), len(sensitive_set_not))
        for S in target_sets:
            for v in v_range:
                S_v = [i for i in S if calibrated_predictions[i]<v+(1./lmbda) and calibrated_predictions[i]>=v]
                print('Cheking bin %.1f ...... size of s_v: '%v, len(S_v))
                if len(S_v)==0:
                    continue
                E_labels = np.mean(lables[S_v])
                # oracle
                E_predictions = np.mean(calibrated_predictions[S_v])
                if abs(E_labels-E_predictions)<alpha/2.:
                    continue
                update = np.random.uniform(E_labels-alpha/4., E_labels+alpha/4.)
                print('Update value: ', update)
                calibrated_predictions[S_v] = calibrated_predictions[S_v] + (update-E_predictions)
                if (calibrated_predictions[S_v]<0).any() or (calibrated_predictions[S_v]>1).any():
                    calibrated_predictions[S_v] = normalize(calibrated_predictions[S_v])
                if set(S_v)!=set([i for i in S if calibrated_predictions[i] < v + (1. / lmbda) and calibrated_predictions[i] >= v]):
                    change += 1
    print(calibrated_predictions)
    print(lables)
    print(predictions)
    print('AE post-calibration: ', abs(np.mean(lables)-np.mean(calibrated_predictions)))


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
    for i in range(len(all_subsets)):
        required_sensitive_features = all_subsets[i]
        #targes_sets[i] = [i for i in range(len(data)) if data[i, sensitive_feature] == 1]
        data_indices_matching_features = [idx for idx in range(len(data)) if 
            data_matches_features(data[idx, :], required_sensitive_features, sensitive_features)]
        target_sets.append(data_indices_matching_features)
    return target_sets
