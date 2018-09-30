#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:30:57 2018
@author: alblle
"""
from alb_MM_functions import Mix_Mod_MethodOfMoments
from SIN_VB_MixMod import Mix_Mod_VB

def Mixture_Model_1Dim(data_vector, opts={'Inference':'Method of moments',
                                        'Number_of_Components':3,'Components_Model':['Gauss','Gamma','-Gamma'],
                                        'init_params':[0,1,3,1,-3,1],'maxits':100,'tol':0.00001}):
    
    if opts['Inference'] == 'Method of moments':
        
        Model = Mix_Mod_MethodOfMoments(data_vector, opts)
        #Model is a dictionary {'means','variances','Mixing Prop.''Likelihood','its','Final responsibilities'}

    elif opts['Inference'] == 'Maximum Likelihood':
        
        print "not implemented yet, very easy... do?"
        Model = 0
        
    elif opts['Inference'] == 'Variational Bayes':

        Model = Mix_Mod_VB(data_vector,opts)
        
    return Model
   
    