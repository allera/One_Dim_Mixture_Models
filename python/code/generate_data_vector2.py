#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:06:19 2018

@author: alblle
"""

def Gamma_shape(m,v):
    shape= (m*m)/v
    return shape

def Gamma_scale(m,v):
    scale= v/m
    return scale

def generate_data_vector(N_comps=3, N_samples=5000 ,means=[0.,5.,-5.], variances= [1.,1.,1.], mixing=[0.8, 0.1, 0.1]):
    
    import numpy as np
    for k in range(N_comps):
        x1=np.random.normal(means[k], variances[k], np.int(np.round(mixing[k] * N_samples)))
        if k==0:
            data_vector=x1
        else:
            data_vector=np.concatenate([data_vector, x1],axis=0)
    #normalize data vector for easier initialization of mixture model fit
    data_vector=np.divide((data_vector-np.mean(data_vector)),np.std(data_vector))
    
    return data_vector

def generate_data_vector_gammas(N_comps=2, N_samples=5000 ,means=[2. ,5.], variances= [2.,2.], mixing=[0.5, 0.5]):
    
    import numpy as np
    for k in range(N_comps):
        
        x1=np.random.gamma(Gamma_shape(means[k], variances[k]), Gamma_scale(means[k], variances[k]) , np.int(np.round(mixing[k] * N_samples)))
        if k==0:
            data_vector=x1
        else:
            data_vector=np.concatenate([data_vector, x1],axis=0)
    #normalize data vector for easier initialization of mixture model fit
    #data_vector=np.divide((data_vector-np.mean(data_vector)),np.std(data_vector))
    
    return data_vector
