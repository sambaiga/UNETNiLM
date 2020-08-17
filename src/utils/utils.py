import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
import random


def get_device():
    if  torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    return device

def set_seed(seed=7777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        



def get_post_neg_weight(labels):
    
    num_postive=[]
    num_negative=[]
    
    total_labels = labels.shape[0]
    for col in range(labels.shape[1]):
        counter=collections.Counter(labels[:,col])
        P      = sum(counter.values())
        num_negative.append(float(counter[0]))
        num_postive.append(float(counter[1]))
        
    post_ratio = np.array(num_postive)/P
    neg_ratio  = np.array(num_negative)/P

    return post_ratio
    

