# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:03:13 2017
@author: allera
"""
import sys
import os
import numpy as np


#Add the toolbox to path
toolbox_path = "/Users/alblle/allera_version_controlled_code/One_Dim_Mixture_Models/python/code"
sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 



import os.path
#load input conenctivity matrix 
#input_file="/Users/alblle/Dropbox/POSTDOC/Demetrius/dmn_non_normalized.csv" 
input_file="/Users/alblle/allera_version_controlled_code/One_Dim_Mixture_Models/python/examples/DMN_net.csv"
#input_file="/Users/alblle/allera_version_controlled_code/One_Dim_Mixture_Models/python/examples/ECN_R_net.csv"
#a1=os.path.isfile(input_file)


#connectivity_matrix = np.loadtxt(input_file, delimiter=',')#, s
#connectivity_matrix =np.genfromtxt(input_file, delimiter=',')


#from Connectome_thresholding_pFDR import GammaGamma_Connectome_thresholding_pFDR
from Connectome_thresholding_pFDR import GaussGammas_Connectome_thresholding_pFDR

threshold=GaussGammas_Connectome_thresholding_pFDR(input_file,toolbox_path)


