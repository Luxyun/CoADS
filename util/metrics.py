# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc

from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

from sksurv.metrics import brier_score,cumulative_dynamic_auc
from sksurv.metrics import concordance_index_censored


def cox_log_rank(hazards, survtime_all, event):
    '''
    :return: the p-value of the logrank test
    '''
    median = np.median(hazards)
    hazard_dichotomize = np.zeros([len(hazards)], dtype=int)
    hazard_dichotomize[hazards >= median] = 1
    idx = hazard_dichotomize == 0
    results = logrank_test(survtime_all[idx], survtime_all[~idx], event_observed_A=event[idx], event_observed_B=event[~idx])
    pvalue=results.p_value
    return pvalue


def get_cindex(hazards, survtime_all, event):
    c_index = concordance_index_censored((event).astype(bool),survtime_all,hazards, tied_tol=1e-08)[0]
    return c_index

def get_mean_std(train_x,valid_x,test_x):
    mean1, mean2, mean3 = map(lambda x: np.mean(x).round(4), [train_x,valid_x,test_x])
    std1, std2, std3 = map(lambda x: np.std(x).round(4), (train_x,valid_x,test_x))
    mean = [mean1, mean2, mean3]
    std = [std1, std2, std3]
    return mean, std