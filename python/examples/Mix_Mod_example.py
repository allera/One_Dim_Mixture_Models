# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:03:13 2017
@author: allera
"""


#Example script to launch fit 1 dimensional mixture model. 


#Add the toolbox to path
import os 
import sys
toolbox_path = "../code" 
sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 


#Generate some data
generate_new_data=1
if generate_new_data==1:
    from generate_data_vector2 import generate_data_vector
    data_vector=generate_data_vector(3, 50000, [0,5,-5], [1,1,1], [0.8, 0.1, 0.1])
else:
    import scipy.io as sio
    sio.loadmat('data_vector.mat')


#Define options
Inference ='Variational Bayes'  #'Method of moments' OR 'Maximum Likelihood' OR 'Variational Bayes' ML NOT INCLUDED YET
Number_of_Components=3
Components_Model=['Gauss','InvGamma','-InvGamma'] #Each component can be Gauss, Gamma, InvGamma, -Gamma, -InvGamma
init_params=[0,1,5,2,-5,2]
maxits=300
tol=0.00000001
opts={'Inference':Inference,'Number_of_Components':Number_of_Components,'Components_Model':Components_Model,
                                        'init_params':init_params,'maxits':maxits,'tol':tol}


from Mixture_Model_1Dim import Mixture_Model_1Dim     
Model = Mixture_Model_1Dim(data_vector, opts)


import matplotlib.pyplot as plt
#plt.plot(Model['Likelihood']);plt.show()
#plt.plot(Model['FEs']);plt.show()
print Model['Mixing Prop.']

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

