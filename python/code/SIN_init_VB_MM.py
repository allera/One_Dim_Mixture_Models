#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:12:22 2018

@author: alblle
"""
from alb_MM_functions import alphaIG,betaIG,alphaGm,betaGm,Mix_Mod_MethodOfMoments
import numpy as np
import copy
import scipy.special as sp
import warnings
warnings.filterwarnings("ignore")

def SIN_init_VB_MM(data,opts):    
    K= opts ['Number_of_Components']
    opts2=copy.deepcopy(opts)
    #opts2['maxits']=1
    
    #SET PRIORS
    #set mixing priors.
    mmm=10;#(the mean of non gauss component)
    vvv=10;#(the variance of the component)\
    m0=np.zeros(K)
    tau0=np.zeros(K)
    b0=np.zeros(K)
    c0=np.zeros(K)
    b_0=np.zeros(K)
    c_0=np.zeros(K)
    d_0=np.zeros(K)
    e_0=np.zeros(K)
    Erate=np.zeros(K)
    Eshape=np.zeros(K)
    loga_0=np.zeros(K)
    Escale=np.zeros(K)
    
    
    for k in range(K):
        if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'): 
        #set GAMMA prior on rates (shape and rate)
            Erate[k]= np.true_divide(1,betaGm(mmm,vvv))
            d_0[k]= copy.deepcopy(Erate[k])
            e_0[k]=1;
            Erate[k]=np.true_divide(d_0[k],e_0[k])
            #set shapes conditional prior (fancy)
            Eshape[k]=alphaGm(mmm,vvv);
            dum_v=np.copy(Eshape[k]);#allow variance on shape to be of size of mean shape
            dum_p=np.true_divide(1,dum_v);
            #from laplace approx b=prec/psi'(map(s))
            b_0[k]=np.true_divide(dum_p,sp.polygamma(1,Eshape[k]))
            c_0[k]=copy.deepcopy(b_0[k]);	
            loga_0[k]=((b_0[k]* sp.polygamma(0,Eshape[k]))-(c_0[k]*np.log(Erate[k])))     
        
        
        if (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'): 
        #set GAMMA prior on scale (shape d and rate e)
            Escale[k]=betaIG(mmm,vvv);
            d_0[k]=copy.deepcopy(Escale[k]);#shape
            e_0[k]=1;#rate
            Escale[k]=np.true_divide(d_0[k],e_0[k]);
                  
            #set component 2 and 3 shape conditional prior (fancy)
            Eshape[k]=alphaIG(mmm,vvv);
            dum_v=np.copy(Eshape[k]);#allow variance on shape to be of size of mean shape
            dum_p=np.true_divide(1,dum_v);
            b_0[k]=np.true_divide(dum_p,sp.polygamma(1,Eshape[k]));#from laplace approx b=prec/psi'(map(s))
            c_0[k]=copy.deepcopy(b_0[k]);
            loga_0[k]=(-(b_0[k]* sp.polygamma(0,Eshape[k]))+(c_0[k]*np.log(Escale[k])))
	
		

    Prior={'lambda_0': 5, 'm_0': 0,  'tau_0': 100,  'c0': 0.001, 'b0': 100 , 'd_0': d_0, 'e_0': e_0 ,
        'loga_0': loga_0, 'b_0': b_0 , 'c_0': c_0 }


#	#SET POSTERIORS initializations using ML mixture models
    
    init_ML=Mix_Mod_MethodOfMoments(data, opts2)
    resp=init_ML['Final responsibilities']
    lambdap = np.sum(resp,1)
    Escales=np.zeros(K)  
    
    for k in range(K):    
        if opts['Components_Model'][k]=='Gauss':        
            m0[k]=init_ML['means'][k]
            tau0[k]=np.mean(np.absolute(init_ML['means']))
            #hyperparam. on precission
            init_prec=np.true_divide(1,init_ML['variances'][k])
            init_var_prec=np.var(np.true_divide(1,init_ML['variances']) , ddof=1)
            c0[k]=alphaGm(init_prec,init_var_prec );#shape
            b0[k]=betaGm(init_prec,init_var_prec );#scale
        if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'):
        ##hyperparam. on rates
            init_rates= np.true_divide(1, betaGm(np.absolute(init_ML['means'][k]),init_ML['variances'][k]))# ,   np.true_divide(1,alb.betaGm(np.absolute(ML_param[4]), ML_param[5]))  ]  ;
            dum_var_r= np.multiply(0.1,init_rates)#(init_rates)* 0.1;#    var(init_rates);
            d_0[k]=alphaGm(init_rates,dum_var_r);#shape
            e_0[k]=np.true_divide(1, betaGm(init_rates,dum_var_r));#rate
            Erate[k]=np.true_divide(d_0[k],e_0[k]) # == init_rates
            
            #hyperparam. on shapes
            init_shapes=alphaGm(np.absolute(init_ML['means'][k]),init_ML['variances'][k])  
            b_0[k]=np.sum(resp[k,:])   
            c_0[k]=b_0[k]
            		#loga_0=((b_0* sp.polygamma(0,init_shapes)-(c_0*log(Erates))); 
            loga_0[k]=np.multiply(b_0[k],sp.polygamma(0,init_shapes)) - (np.multiply(c_0[k],np.log(Erate[k])))
            #MAP_shapes=invpsi((loga_0+ (c_0 .* log(Erates))) ./ b_0) # == init_shapes
        if (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'): 
        #hyperparam. on scales (inverse gamma) --> scale is r in the text, #r ~ inv gamma distr
            init_scales=betaIG(np.absolute(init_ML['means'][k]),init_ML['variances'][k])
            dum_var_sc=np.multiply(0.1,init_scales)
            d_0[k]=alphaGm(init_scales,dum_var_sc) #gamma shape
            e_0[k]=np.divide(1., betaGm(init_scales,dum_var_sc)) #gamma rate
            Escales[k]=np.divide(d_0[k],e_0[k])
            #hyperparam. on shapes
            init_shapes=alphaIG(np.absolute(init_ML['means'][k]),init_ML['variances'][k])
            sumgam=np.sum(resp,1) 
            b_0[k]=sumgam[k]
            c_0[k]=copy.deepcopy(b_0[k])  
            loga_0[k]=-np.multiply(b_0[k],sp.polygamma(0,init_shapes)) + (np.multiply(c_0[k],np.log(Escales[k])))
            #MAP_shapes=invpsi((-loga_0+ (c_0 .* log(Escales))) ./ b_0) # == init_shapes

	
    shapes=np.zeros(K)
    rates=np.zeros(K)
    scales=np.zeros(K)
    for k in range(K):                        	
        if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'):
            
            shapes[k]=alphaGm(np.absolute(init_ML['means'][k]),init_ML['variances'][k])
            rates[k]=np.divide( 1,  betaGm(np.absolute(init_ML['means'][k]),init_ML['variances'][k]))
            
        if (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):
            
            shapes[k]=alphaIG(np.absolute(init_ML['means'][k]),init_ML['variances'][k])
            scales[k]=  betaIG(np.absolute(init_ML['means'][k]),init_ML['variances'][k])
            
    #Save posterior expectations for initialization of VB mixModel        
    Posterior={'gammas': resp,'pi': init_ML['Mixing Prop.'],'mus': init_ML['means'],
        'tau1s':np.true_divide(1, init_ML['variances']),'shapes': shapes,'rates': rates,'scales': scales,
          'lambda': lambdap, 'm_0': m0,  'tau_0': tau0,  'c0': c0, 'b0': b0 ,
       'd_0': d_0, 'e_0': e_0 , 'loga_0': loga_0, 'b_0': b_0 , 'c_0': c_0 }
	 
            
    return Prior, Posterior