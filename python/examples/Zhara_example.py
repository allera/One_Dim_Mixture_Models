#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:41:52 2019

@author: alblle
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:17:25 2019

@author: alblle
"""

#Example script to launch fit 1 dimensional mixture model. 


#Add the toolbox to path
import os 
import sys
toolbox_path = "../code" 
sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 
#Add the toolbox to path


#Generate some data
generate_new_data=1
if generate_new_data==1:
    from generate_data_vector2 import generate_data_vector
    data_vector=generate_data_vector(N_comps=3, N_samples=5000 ,means=[ 0., 5. ,-5.], variances= [1.,2.,2.], mixing=[0.9, 0.09, 0.01])
else:
    import scipy.io as sio
    sio.loadmat('data_vector.mat')
#Generate some data
    

#Define options for the mixture model fit
Inference ='Variational Bayes'#'Method of moments'#'Variational Bayes'  #'Method of moments' OR 'Maximum Likelihood' OR 'Variational Bayes' ML NOT INCLUDED YET
Number_of_Components=3
Components_Model=['Gauss','Gamma','-Gamma'] #Each component can be Gauss, Gamma, InvGamma, -Gamma, -InvGamma
init_params=[0.,1.,6.,1.,-6.,1.]
init_pi=np.ones(3);
init_pi=np.divide(init_pi,3)
maxits=300
tol=0.00000001
opts={'Inference':Inference,'Number_of_Components':Number_of_Components,'Components_Model':Components_Model,
                                        'init_params':init_params,'maxits':maxits,'tol':tol,'init_pi':init_pi}
#Define options for the mixture model fit


# CALL TO FIT MIXTURE MODEL
from Mixture_Model_1Dim import Mixture_Model_1Dim     
Model = Mixture_Model_1Dim(data_vector, opts)
print Model['Mixing Prop.']
# CALL TO FIT MIXTURE MODEL




# Plot the resulting fit on a histogram of the data
import numpy as np
from alb_MM_functions import invgam
from alb_MM_functions import gam
from scipy.stats import norm

my_range=np.linspace(0.001,10,10000)

#plt0=np.multiply( Model['Mixing Prop.'][0],norm.pdf(my_range,Model['mu1'][0],np.sqrt(np.divide(1,Model['tau1s'][0]))  ) )
dist_plt=np.zeros([Number_of_Components,10000])
for k in range(Number_of_Components):
    if Components_Model[k]=='InvGamma':
        dist_plt[k,:]=np.multiply( Model['Mixing Prop.'][k],invgam(my_range,Model['shapes'][k],Model['scales'][k]))
    elif Components_Model[k]=='Gamma':
        dist_plt[k,:]=np.multiply( Model['Mixing Prop.'][k],gam(my_range,Model['shapes'][k],np.divide(1,Model['rates'][k])))
    #elif Components_Model[k]=='-InvGamma':
    #    dist_plt[k,:]=np.multiply( Model['Mixing Prop.'][k],invgam(-my_range,Model['shapes'][k],Model['scales'][k]))
    #elif Components_Model[2]=='-Gamma':
    #    dist_plt[k,:]=np.multiply( Model['Mixing Prop.'][k],gam(-my_range,Model['shapes'][k],np.divide(1,Model['rates'][k])))
        
#plt[:,my_range>0]=0
full_fit=np.sum(dist_plt,0)
        
    


import matplotlib.pyplot as plt
plt.hist(data_vector,bins=50,density=True,alpha=1, color='g')
#plt.plot(my_range,plt0, 'k', linewidth=2)
for k in range(Number_of_Components):
    plt.plot(my_range,dist_plt[k,:], 'k', linewidth=2)

plt.plot(my_range,full_fit, 'r', linewidth=2)
plt.show()