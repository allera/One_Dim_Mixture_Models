# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:03:13 2017
@author: allera
"""
import sys
import os


#Add the toolbox to path
toolbox_path = "/Users/alblle/allera_version_controlled_code/One_Dim_Mixture_Models/python/code"
sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 




#load input conenctivity matrix 
input_file="/Users/alblle/Dropbox/POSTDOC/Demetrius/dmn_non_normalized.csv" 

from GammaGamma_Connectome_thresholding_pFDR import GammaGamma_Connectome_thresholding_pFDR
threshold=GammaGamma_Connectome_thresholding_pFDR(input_file,toolbox_path)


