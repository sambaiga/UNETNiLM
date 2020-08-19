from sklearn import metrics
import math
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from collections import OrderedDict



def get_mae(target, prediction):
    return mean_absolute_error(target, prediction)

def get_eac(target, prediction):
    num = np.sum(np.abs(prediction-target))
    den = (np.sum(target))
    eac = 1 - (num/den)/2
    return np.where(eac<0, 0, eac)

def get_relative_error(target, prediction):
    return np.mean(np.nan_to_num(np.abs(target - prediction) / np.maximum(target, prediction)))


def get_nde(target, prediction):
    return np.sum((target - prediction) ** 2) / np.sum((target ** 2))

def get_sae(target, prediction, sample_second):
    '''
    compute the signal aggregate error
    sae = |\hat(r)-r|/r where r is the ground truth total energy;
    \hat(r) is the predicted total energy.
    '''
    r = np.sum(target * sample_second * 1.0 / 3600.0)
    rhat = np.sum(prediction * sample_second * 1.0 / 3600.0)
    return np.abs(r - rhat) / np.abs(r)

def subset_accuracy(true_targets, predictions, per_sample=False, axis=1):
    result = np.all(true_targets == predictions, axis=axis)
    if not per_sample:
        result = np.mean(result)

    return result

def compute_jaccard_score(true_targets, predictions, per_sample=False, average='macro'):
    if per_sample:
        jaccard = metrics.jaccard_score(true_targets, predictions, average=None)
    else:
        if average not in set(['samples', 'macro', 'weighted']):
            raise ValueError("Specify samples or macro")
        jaccard = metrics.jaccard_score(true_targets, predictions, average=average)
    return jaccard

def hamming_loss(true_targets, predictions, per_sample=False, axis=1):

    result = np.mean(np.logical_xor(true_targets, predictions),
                        axis=axis)

    if not per_sample:
        result = np.mean(result)

    return result


def compute_tp_fp_fn(true_targets, predictions, axis=1):
    # axis: axis for instance
    tp = np.sum(true_targets * predictions, axis=axis).astype('float32')
    fp = np.sum(np.logical_not(true_targets) * predictions,
                   axis=axis).astype('float32')
    fn = np.sum(true_targets * np.logical_not(predictions),
                   axis=axis).astype('float32')

    return (tp, fp, fn)


def example_f1_score(true_targets, predictions, per_sample=False, axis=1):
    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)

    numerator = 2*tp
    denominator = (np.sum(true_targets,axis=axis).astype('float32') + np.sum(predictions,axis=axis).astype('float32'))

    zeros = np.where(denominator == 0)[0]

    denominator = np.delete(denominator,zeros)
    numerator = np.delete(numerator,zeros)

    example_f1 = numerator/denominator


    if per_sample:
        f1 = example_f1
    else:
        f1 = np.mean(example_f1)

    return f1

def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    if average == 'micro':
        f1 = 2*np.sum(tp) / \
            float(2*np.sum(tp) + np.sum(fp) + np.sum(fn))

    elif average == 'macro':

        def safe_div(a, b):
            """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(a, b)
            return c[np.isfinite(c)]

        f1 = np.mean(safe_div(2*tp, 2*tp + fp + fn))

    return f1

def f1_score(true_targets, predictions, average='micro', axis=1):
    """
        average: str
            'micro' or 'macro'
        axis: 0 or 1
            label axis
    """
    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)
    f1 = f1_score_from_stats(tp, fp, fn, average=average)

    return f1


def compute_metrics(all_predictions,all_targets,all_metrics=True,verbose=False):
        
    acc_ = list(subset_accuracy(all_targets, all_predictions, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions, axis=0, per_sample=True))
    exjacc_ = list(compute_jaccard_score(all_targets, all_predictions, per_sample=True))
          
    acc = round(np.mean(acc_), 4)
    hl = round(np.mean(hl_), 4)
    exf1 = round(np.mean(exf1_), 4)
    per_app_fscore = exf1_
    

    tp, fp, fn = compute_tp_fp_fn(all_targets, all_predictions, axis=0)
    mif1 = round(f1_score_from_stats(tp, fp, fn, average='micro'),4)
    maf1 = round(f1_score_from_stats(tp, fp, fn, average='macro'),4)
    
    eval_ret = OrderedDict([('Subset accuracy', acc),
                        ('Hamming accuracy', 1 - hl),
                        ('Example-based F1', exf1),
                        ('Label-based Micro F1', mif1),
                        ('Label-based Macro F1', maf1)])

    
    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    miF1 = eval_ret['Label-based Micro F1']
    maF1 = eval_ret['Label-based Macro F1']
    if verbose:
        print('ACC:   '+str(ACC))
        print('HA:    '+str(HA))
        print('ebF1:  '+str(ebF1))
        print('miF1:  '+str(miF1))
        print('maF1:  '+str(maF1))
        
        
    
    metrics_dict = {}
    metrics_dict['subACC'] = ACC
    #metrics_dict['JACC'] = jacc
    #metrics_dict['exACC'] = acc_
    metrics_dict['appF1'] = per_app_fscore
    #metrics_dict['exHA'] = hl_
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['miF1'] = miF1
    metrics_dict['maF1'] = maF1
    return metrics_dict


def compute_regress_metrics(target, prediction):
    eac = get_eac(target, prediction)
    mae = get_mae(target, prediction)
    nade = get_nde(target, prediction)
    
   
    metrics = OrderedDict([('EAC', eac),
                        ('MAE', mae),
                        ('NDE', nade)])
    
    metrics_dict = {}
    metrics_dict['EAC'] = metrics["EAC"]
    metrics_dict['MAE'] = metrics["MAE"]
    metrics_dict['NDE'] = metrics["NDE"]
    
    return metrics_dict
