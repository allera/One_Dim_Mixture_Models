# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:03:13 2017
@author: allera
"""


import sys
import os
import numpy as np
import nibabel as nib
import time


#Add the toolbox to path
toolbox_path = "/Users/alblle/allera_version_controlled_code/One_Dim_Mixture_Models/python/code"
sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 
from Mixture_Model_1Dim import Mixture_Model_1Dim
Inferences_possibilities=['Method of moments','Maximum Likelihood','Variational Bayes']    
     


data_path='/Users/alblle/Dropbox/POSTDOC/MYPAPERS/OHBM_journal_2020_mix_models_paper/new_code/Raimon_templates/Raipru_11_ICs_2mm.nii.gz' 



#wd=os.getcwd()
#completeName = os.path.join(wd, 'Results', "OUTPUT_IMAGENAME") #name of directory to save results"
#completeName = "OUTDIR"

#os.mkdir(completeName)
img = nib.load(data_path)
data = img.get_data()

origsize=data.shape
IC=np.reshape(data,[data.shape[0]* data.shape[1]* data.shape[2],data.shape[3]], "F" )
numVoxels=IC.shape[0]
numICs=IC.shape[1]


N_models=2 #GGG, GII
N_Inference_pos=len(Inferences_possibilities)



MixProp=np.zeros([numICs,N_models,N_Inference_pos,3]) 
COST=np.zeros([numICs,N_models,N_Inference_pos]);# 6 models
Threshold=np.zeros([numICs,N_models,N_Inference_pos,2]);# 6 models, 2 thresholds per model
Its=np.zeros([numICs,N_models,N_Inference_pos])


#mixture options
init_params=[0,1,4,2,-4,2]

maxits=300
tol=0.000001


#basis_dict={MixProp}
#Results={'Gauss_Gamma':}
if 1:

    for icnumber in range (0,numICs): 
        myIC=IC[:,icnumber]
        x=np.copy(myIC)
        
        
        NoBrainVoxels=np.argwhere(x==0)
        BrainVoxels=np.argwhere(x!=0)
        #remove zero voxels from x
        all_x=x;#all voxels
        PICAMAP=np.zeros(all_x.shape[0])
        x=x[BrainVoxels]#maxed voxels
        x=np.squeeze(np.divide(x-x.mean(),x.std()))
    
        
        
        # Gauss Gamma-Gamma
        Number_of_Components=3
        init_pi=np.divide(np.ones(Number_of_Components),Number_of_Components)
        Components_Model_types=[['Gauss','Gamma','-Gamma'],['Gauss','InvGamma','-InvGamma']] #Each component can be Gauss, Gamma, InvGamma, -Gamma, -InvGamma
        
        for dist_type in range(N_models):
            
            Components_Model=Components_Model_types[dist_type]
            
            for inference_type in range(N_Inference_pos):
                
                Inference =Inferences_possibilities[inference_type]
                
                opts={'Inference':Inference,'Number_of_Components':Number_of_Components,'Components_Model':Components_Model,
                                'init_params':init_params,'maxits':maxits,'tol':tol,'init_pi':init_pi}
                
                t = time.time()
                Model = Mixture_Model_1Dim(x, opts)
                
                
                COST[icnumber,dist_type,inference_type] = time.time() - t
                MixProp[icnumber,dist_type,inference_type,:]= Model['Mixing Prop.']
                Its[icnumber,dist_type,inference_type]=Model['its']
                
                resp=np.squeeze(np.asarray(Model['Final responsibilities']))
                qq=resp[0,:]
                qq[qq>0.5]=1
                qq[resp[1,:]>0.5]=2
                if np.sum(qq==2)>0:
                    Threshold[icnumber,dist_type,inference_type,0]=x[qq==2].min()
                else:
                    Threshold[icnumber,dist_type,inference_type,0]=x.max()
    
                if Number_of_Components==3:
                    qq[resp[2,:]>0.5]=3
                    if np.sum(qq==3)>0:
                        Threshold[icnumber,dist_type,inference_type,1]=x[qq==3].max()
                    else:
                        Threshold[icnumber,dist_type,inference_type,1]=x.min()
    
    
    
                
                print(Components_Model)
                print(Inference)
                print(COST[icnumber,dist_type,inference_type])
                print(MixProp[icnumber,dist_type,inference_type,:])
                #print(Its[icnumber,dist_type,inference_type])
                #print(Threshold[icnumber,dist_type,inference_type,:])
    
            #	find classifiers threshodls
    #        ng=np.where(src1['q'] ==2);
    #        ps=np.where(src1['q'] ==1);
    #        if np.size(ps)!=0:
    #            Threshold[icnumber][0][0]=np.min(x[ps])
    #        if np.size(ng)!=0:
    #            Threshold[icnumber][0][1]=np.max(x[ng])
    
    
for dist_type in range(N_models):
        
        
        for inference_type in range(N_Inference_pos):
            
            print(COST[:,dist_type,inference_type])   
            print(MixProp[:,dist_type,inference_type,:])

np.mean(COST,0)
np.nanmean(MixProp,0)
#savedfile= os.path.join(completeName,'RESULTS.mat')
#Results=[]
#Results.append({'Iterations': Its, 'MixingProps' : MixProp, 'Thresholds' :Threshold, 'cost':COST})
#sio.savemat(savedfile,{'Results':Results })    
    
    
    


	




