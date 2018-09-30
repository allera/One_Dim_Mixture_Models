#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:06:19 2018

@author: alblle
"""
def generate_data_vector(d):
    
    import numpy as np
    s=d
    N_samples=10000
    x1 = np.random.normal(0, 1, np.int(np.round(0.8 * N_samples)));
    x2=  np.random.normal(5, 1, np.int(np.round(0.1 * N_samples)));
    x3=  np.random.normal(-5, 1, np.int(np.round(0.1 * N_samples)));
    
    data_vector=np.concatenate([x1, x2, x3],axis=0)
    #normalize data vector for easier initialization
    data_vector=np.divide((data_vector-np.mean(data_vector)),np.std(data_vector))
    
    return data_vector
