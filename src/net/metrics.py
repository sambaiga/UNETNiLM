from sklearn import metrics
import math
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from collections import OrderedDict
import pandas as pd

def get_mae(target, prediction):
    return np.abs(target - prediction).mean(axis=0)

def get_eac(target, prediction):
     num = np.abs(target - prediction).sum(axis=0)
     den = 2*target.sum(axis=0)
     return (1 - num/den)

def get_nde(target, prediction):
    return np.sum((target - prediction) ** 2) / np.sum((target ** 2), axis=0) 

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

def compute_metrics(y_t, y_p):
    tp, fp, fn = compute_tp_fp_fn(y_t, y_p, axis=0)
    mif1 = round(f1_score_from_stats(tp, fp, fn, average='micro'),4)
    maf1 = round(f1_score_from_stats(tp, fp, fn, average='macro'),4)
    hl_ = hamming_loss(y_t, y_p, axis=0, per_sample=True)
    exf1_ = list(example_f1_score(y_t, y_p, axis=0, per_sample=True))
    hl = round(np.mean(hl_), 4)
    exf1 = round(np.mean(exf1_), 4)
    
    metrics_dict = {}
    metrics_dict['appF1'] = exf1_ 
    metrics_dict['HA'] = hl_
    metrics_dict['ebF1'] = exf1
    metrics_dict['miF1'] = mif1
    metrics_dict['maF1'] = maf1
    return metrics_dict


def compute_regress_metrics(y_t, y_p):
    eac = get_eac(y_t, y_p)
    nde = get_nde(y_t, y_p)
    mae = get_mae(y_t, y_p)
    metrics_dict = {}
    metrics_dict['EAC'] = eac
    metrics_dict['NDE'] = nde
    metrics_dict['MAE'] = mae
    return metrics_dict

def get_results_summary(z_t, z_p, y_t, y_p, appliances, data="UKDALE"):
    
    reg = compute_regress_metrics(y_t, y_p)
    mlb = compute_metrics(z_t, z_p)
    
    per_app = {'EAC': reg['EAC'].tolist(),
          'NDE': reg['NDE'].tolist(),
          'MAE': reg['MAE'].tolist(),
          'exbF1': mlb['appF1'],
          'HA': (1-mlb['HA']).tolist()}
    per_app =pd.DataFrame.from_dict(per_app, orient="index")
    per_app.columns = appliances
    avg_results = {'EAC': reg['EAC'].mean().tolist(),
          'NDE': reg['NDE'].mean().tolist(),
          'MAE': reg['MAE'].mean().tolist(),
          'exbF1': mlb['ebF1'].tolist(),
           'maF1': mlb['maF1'].tolist(),
           'miF1': mlb['miF1'].tolist(),
          'HA': (1-mlb['HA']).mean().tolist()}
    avg_results =pd.DataFrame.from_dict(avg_results, orient="index")
    avg_results.columns = [data]
    return per_app, avg_results



