#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:52:11 2020

@author: alblle
"""


#Example script to launch fit 1 dimensional mixture model. 


#Add the toolbox to path
import os 
import sys
import numpy as np
toolbox_path = "/Users/alblle/allera_version_controlled_code/One_Dim_Mixture_Models/python/code"# ../code" 
sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 
#Add the toolbox to path


#Generate some data
generate_new_data=0
if generate_new_data==1:
    from generate_data_vector2 import generate_data_vector
    data_vector=generate_data_vector(3, 50000, [0,5,-5], [1,1,1], [0.8, 0.1, 0.1])
else:
    #import scipy.io as sio
    #sio.loadmat('data_vector.mat')
    data_vector=np.loadtxt('/Users/alblle/allera_version_controlled_code/One_Dim_Mixture_Models/python/examples/data_vector_overflowerror.txt')

#Generate some data
    

#Define options for the mixture model fit
Inference ='Method of moments'#'Variational Bayes'#'Method of moments'#'Variational Bayes'  #'Method of moments' OR 'Maximum Likelihood' OR 'Variational Bayes' ML NOT INCLUDED YET
Number_of_Components=3
Components_Model=['Gauss','InvGamma','-InvGamma'] #Each component can be Gauss, Gamma, InvGamma, -Gamma, -InvGamma
init_params=[0,1,5,2,-5,2]
init_pi=np.ones(3);
init_pi=np.divide(init_pi,3)
#init_pi[0]=0.9;init_pi[1]=0.05;init_pi[2]=0.05
maxits=300
tol=0.00000001
opts={'Inference':Inference,'Number_of_Components':Number_of_Components,'Components_Model':Components_Model,
                                        'init_params':init_params,'maxits':maxits,'tol':tol,'init_pi':init_pi}
#Define options for the mixture model fit


# CALL TO FIT MIXTURE MODEL
from Mixture_Model_1Dim import Mixture_Model_1Dim     
Model = Mixture_Model_1Dim(data_vector, opts)
#print Model['Mixing Prop.']
# CALL TO FIT MIXTURE MODEL



# Plot the resulting fit on a histogram of the data
import numpy as np
from alb_MM_functions import invgam
from alb_MM_functions import gam
from scipy.stats import norm

my_range=np.linspace(-10,10,10000)

plt0=np.multiply( Model['Mixing Prop.'][0],norm.pdf(my_range,Model['mu1'][0],np.sqrt(np.divide(1,Model['taus1'][0]))  ) )

if Components_Model[1]=='InvGamma':
    plt1=np.multiply( Model['Mixing Prop.'][1],invgam(my_range,Model['shapes'][1],Model['scales'][1]))
elif Components_Model[1]=='Gamma':
    plt1=np.multiply( Model['Mixing Prop.'][1],gam(my_range,Model['shapes'][1],np.divide(1,Model['rates'][1])))
    
plt1[my_range<0]=0

if Components_Model[2]=='-InvGamma':
    plt2=np.multiply( Model['Mixing Prop.'][2],invgam(-my_range,Model['shapes'][2],Model['scales'][2]))
elif Components_Model[2]=='-Gamma':
    plt2=np.multiply( Model['Mixing Prop.'][2],gam(-my_range,Model['shapes'][2],np.divide(1,Model['rates'][2])))
    
plt2[my_range>0]=0


plt1=np.nan_to_num(plt1)

import matplotlib.pyplot as plt
plt.hist(data_vector,bins=50,density=True,alpha=1, color='g')
plt.plot(my_range,plt0, 'k', linewidth=2)
plt.plot(my_range,plt1, 'k', linewidth=2)
plt.plot(my_range,plt2, 'k', linewidth=2)
plt.plot(my_range,plt0+plt1+plt2, 'r', linewidth=2)
plt.show()





# Plot the resulting fit on a histogram of the data



#def plot_fit(data_vector,opts,Model):
#    #import matplotlib.pyplot as plt
#    if opts['Inference']= 'Method of moments':
#        
#    elif opts['Inference']= 'Maximum Likelihood':
#        
#    elif opts['Inference']= 'Variational Bayes':
#    
#    
#    return 

