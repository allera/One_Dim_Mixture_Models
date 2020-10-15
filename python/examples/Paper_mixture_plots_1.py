#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:02:06 2020

@author: alblle
"""

mycolor=[(225./255,141./255,16./255)]


#Add the toolbox to path
import os 
import sys
import numpy as np
import scipy

from scipy.stats import norm
import matplotlib.pyplot as plt




toolbox_path = "/Users/alblle/allera_version_controlled_code/One_Dim_Mixture_Models/python/code" 
sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 
from Mixture_Model_1Dim import Mixture_Model_1Dim   
from alb_MM_functions import alphaGm
from alb_MM_functions import betaGm
from alb_MM_functions import invgam
from alb_MM_functions import gam





fig, ax = plt.subplots(1, 3,figsize=(9, 3))#, sharey=True) 
#axs[0].plot(x, y)
#axs[1].scatter(x, y)  

#Add the toolbox to path


#GAUSS GAMMAS PLOT
N_samples=10000
x1 = np.random.normal(0, 2, np.int(np.round(0.5 * N_samples)));
x2=  np.random.normal(5, 3, np.int(np.round(0.25 * N_samples)));
x3=  np.random.normal(-5, 3, np.int(np.round(0.25 * N_samples)));
data_vector=np.concatenate([x1, x2, x3],axis=0)
#normalize data vector for easier initialization
#data_vector=np.divide((data_vector-np.mean(data_vector)),np.std(data_vector))


#Define options for the mixture model fit
Components_Model=['Gauss','Gamma','-Gamma']
opts={'Inference':'Maximum Likelihood','Number_of_Components':3,'Components_Model':['Gauss','Gamma','-Gamma'],
                                        'init_params':[0,1,5,2,-5,2],'maxits':300,'tol':0.00000001,'init_pi':np.divide(np.ones(3),3)}
# CALL TO FIT MIXTURE MODEL
Model = Mixture_Model_1Dim(data_vector, opts)

# Plot the resulting fit on a histogram of the data


T=10000
my_range=np.linspace(-15,15,T)
PLTS=np.zeros([3,T])


for k in range(3):
    if Components_Model[k]=='Gauss':
        PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],norm.pdf(my_range,Model['mu1'][k],np.sqrt(np.divide(1,Model['taus1'][k]))  ) )
        
    elif Components_Model[k]=='InvGamma':
        PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],invgam(my_range,Model['shapes'][k],Model['scales'][k]))
        PLTS[k,my_range<0]=0

    elif Components_Model[k]=='Gamma':
        PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],gam(my_range,Model['shapes'][k],np.divide(1,Model['rates'][k])))
        PLTS[k,my_range<0]=0

    elif Components_Model[k]=='-InvGamma':
         PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],invgam(-my_range,Model['shapes'][k],Model['scales'][k]))
         PLTS[k,my_range>0]=0

    elif Components_Model[k]=='-Gamma':
         PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],gam(-my_range,Model['shapes'][k],np.divide(1,Model['rates'][k])))
         PLTS[k,my_range>0]=0

   
        
ax[0].hist(data_vector,bins=50,density=True,alpha=0.7, color=mycolor)
for k in range(3):
    ax[0].plot(my_range, PLTS[k,:], 'k', linewidth=2)

ax[0].plot(my_range,np.sum(PLTS,0), 'r', linewidth=2)
#plt.show()  

ax[0].set_ylim(0,0.15) 
ax[0].set_yticks([0,0.05,0.1,0.15])
ax[0].set_yticklabels([0,0.05,0.1,0.15]) 

ax[0].set_xlim(-12,12) 
ax[0].set_xticks([-10,-5,0,5,10]) 
ax[0].set_xticklabels([-10,-5,0,5,10]) 

ax[0].grid('True')

 


del PLTS


#GAMMAS PLOT

y1 = np.random.gamma(2, 2,int(0.75*N_samples))
y2 = np.random.gamma(10, 2,int(0.25*N_samples))
data_vector=np.concatenate([y1,y2],axis=0)
#Define options for the mixture model fit
Components_Model=['Gamma','Gamma']
opts={'Inference':'Maximum Likelihood','Number_of_Components':2,'Components_Model':['Gamma','Gamma'],
                                        'init_params':[2,5,20,5],'maxits':300,'tol':0.00000001,'init_pi':np.divide(np.ones(2),2)}
# CALL TO FIT MIXTURE MODEL
Model = Mixture_Model_1Dim(data_vector, opts)


my_range=np.linspace(0.001,30,T)
PLTS=np.zeros([2,T])

for k in range(2):
    if Components_Model[k]=='Gauss':
        PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],norm.pdf(my_range,Model['mu1'][k],np.sqrt(np.divide(1,Model['taus1'][k]))  ) )
        
    elif Components_Model[k]=='InvGamma':
        PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],invgam(my_range,Model['shapes'][k],Model['scales'][k]))
        PLTS[k,my_range<0]=0

    elif Components_Model[k]=='Gamma':
        PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],gam(my_range,Model['shapes'][k],np.divide(1,Model['rates'][k])))
        PLTS[k,my_range<0]=0

    elif Components_Model[k]=='-InvGamma':
         PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],invgam(-my_range,Model['shapes'][k],Model['scales'][k]))
         PLTS[k,my_range>0]=0

    elif Components_Model[k]=='-Gamma':
         PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],gam(-my_range,Model['shapes'][k],np.divide(1,Model['rates'][k])))
         PLTS[k,my_range>0]=0
         
         
ax[1].hist(data_vector,bins=50,density=True,alpha=.7, color=mycolor)
for k in range(2):
    ax[1].plot(my_range, PLTS[k,:], 'k', linewidth=2)

ax[1].plot(my_range,np.sum(PLTS,0), 'r', linewidth=2)


ax[1].set_ylim(0,0.15) 
ax[1].set_yticks([0,0.05,0.1,0.15])
#ax[1].set_yticklabels([0,0.05,0.1,0.15]) 
ax[1].set_yticklabels([]) 


ax[1].set_xlim(0,30) 
ax[1].set_xticks([0,10,20,30]) 
ax[1].set_xticklabels([0,10,20,30]) 

ax[1].grid('True')



del PLTS

#BETAS PLOT

# parameters for generating synthetic data
N_total=10000 #total number of samples
mix=np.array([.75, .25])  #mix proportions
a1=1;b1=5;
a2=20;b2=5;


#generate the data
sample_sizes=np.floor(N_total*mix);
x1=np.random.beta(a1,b1,int(sample_sizes[0]))
x2=np.random.beta(a2,b2,int(sample_sizes[1]))
data_vector=np.append(x1, x2)

init_means=np.multiply(range(2),np.divide(np.ones(2),2))
init_means=init_means+np.divide(np.ones(2),2*2)

init_params=0.01*(np.ones(2*2))
for k in range(2):
    init_params[2*k]=init_means[k]
    
Components_Model=['Beta','Beta']
    
opts={'Inference':'Maximum Likelihood','Number_of_Components':2,'Components_Model':['Beta','Beta'],
                                        'init_params':init_params,'maxits':300,'tol':0.00000001,'init_pi':np.divide(np.ones(2),2)}

Model = Mixture_Model_1Dim(data_vector, opts)

PLTS=np.zeros([2,T])
my_range=np.linspace(0,1,T)
PLTS=np.zeros([2,T])


ax[2].hist(data_vector,bins=50,density=True,alpha=.7, color=mycolor)
for k in range(2):
    PLTS[k,:]=np.multiply( Model['Mixing Prop.'][k],scipy.stats.beta.pdf(my_range,Model['shapes'][k],Model['scales'][k])  ) 
    ax[2].plot(my_range,PLTS[k,:], 'k', linewidth=2)
    
ax[2].plot(my_range,np.sum(PLTS,axis=0), 'r', linewidth=2)


ax[2].set_ylim(0,3.6) 
ax[2].set_yticks([0,1,2,3])
ax[2].set_yticklabels([0,1,2,3]) 

ax[2].set_xlim(0,1) 
ax[2].set_xticks([0,0.5,1]) 
ax[2].set_xticklabels([0,0.5,1]) 

ax[2].grid('True')


plt.savefig('mixtures.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


#plt.show()
1
    