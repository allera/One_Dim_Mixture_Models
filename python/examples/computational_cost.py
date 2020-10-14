#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:51:35 2020

@author: alblle
"""

#Example script to launch fit 1 dimensional mixture model. 


#Add the toolbox to path
import time

import os 
import sys
import numpy as np
toolbox_path = "../code" 
sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 
#Add the toolbox to path


#Generate some data
generate_new_data=1
if generate_new_data==1:
    from generate_data_vector2 import generate_data_vector
    data_vector=generate_data_vector(3, 50000, [0,3,-3], [1,1,1], [0.9, 0.05, 0.05])
else:
    import scipy.io as sio
    sio.loadmat('data_vector.mat')
#Generate some data
    

#Define options for the mixture model fit
Inferences_possibilities=['Method of moments','Maximum Likelihood','Variational Bayes']    
Number_of_Components=3
 #Each component can be Gauss, Gamma, InvGamma, -Gamma, -InvGamma
init_params=[0,1,5,2,-5,2]
init_pi=np.ones(3);
init_pi=np.divide(init_pi,3)
maxits=300
tol=0.00000001
opts={'Inference':Inference,'Number_of_Components':Number_of_Components,'Components_Model':Components_Model,
                                        'init_params':init_params,'maxits':maxits,'tol':tol,'init_pi':init_pi}


# CALL TO FIT MIXTURE MODEL
from Mixture_Model_1Dim import Mixture_Model_1Dim  

Components_Model=['Gauss','Gamma','-Gamma']
for inference in range(3):
    Inference =Inferences_possibilities[inference]  
    t = time.time()
    Model = Mixture_Model_1Dim(data_vector, opts)
    elapsed = time.time() - t
    
    print(elapsed)
    print(Model['its'])
    
Components_Model=['Gauss','InvGamma','-InvGamma']
for inference in range(3):
    Inference =Inferences_possibilities[inference]  
    t = time.time()
    Model = Mixture_Model_1Dim(data_vector, opts)
    elapsed = time.time() - t
    
    print(elapsed)   
    print(Model['its'])

    

#print Model['Mixing Prop.']
# CALL TO FIT MIXTURE MODEL


plotme=0
if plotme:
    # Plot the resulting fit on a histogram of the data
    import numpy as np
    from alb_MM_functions import invgam
    from alb_MM_functions import gam
    from scipy.stats import norm
    
    T=10000
    my_range=np.linspace(-10,10,T)
    PLTS=np.zeros([Number_of_Components,T])
    
    
    for k in range(Number_of_Components):
        if Components_Model[k]=='Gauss':
            PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],norm.pdf(my_range,Model['mu1'][k],np.sqrt(np.divide(1,Model['taus1'][k]))  ) )
            
        elif Components_Model[k]=='InvGamma':
            PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],invgam(my_range,Model['shapes'][k],Model['scales'][k]))
            PLTS[k,my_range<0]=0
    
        elif Components_Model[k]=='Gamma':
            PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],gam(my_range,Model['shapes'][k],np.divide(1,Model['rates'][k])))
            PLTS[k,my_range<0]=0
    
        elif Components_Model[2]=='-InvGamma':
             PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],invgam(-my_range,Model['shapes'][k],Model['scales'][k]))
             PLTS[k,my_range>0]=0
    
        elif Components_Model[2]=='-Gamma':
             PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],gam(-my_range,Model['shapes'][k],np.divide(1,Model['rates'][k])))
             PLTS[k,my_range>0]=0
    
       
            
    import matplotlib.pyplot as plt
    plt.hist(data_vector,bins=50,density=True,alpha=1, color='g')
    for k in range(Number_of_Components):
        plt.plot(my_range, PLTS[k,:], 'k', linewidth=2)
    
    plt.plot(my_range,np.sum(PLTS,0), 'r', linewidth=2)
    plt.show()   
    
    
    plt.clf()
    plt.plot(Model['Likelihood'])   
        
        
            
