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
Inferences_possibilities=['Variational Bayes'] #['Method of moments']#,'Maximum Likelihood','Variational Bayes']    
     


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






#mixture options
init_params=[0,1,5,2,-5,2]

maxits=300
tol=0.000001


#for icnumber in range (0,numICs):
icnumber=1
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
        
        
        COST = time.time() - t
        MixProp= Model['Mixing Prop.']
        Its=Model['its']
        
        


        print(icnumber)
        print(Components_Model)
        print(Inference)
        print(COST)
        print(MixProp)
       