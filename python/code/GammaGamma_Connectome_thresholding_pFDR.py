import numpy as np
import sys
import os
import scipy


def GammaGamma_Connectome_thresholding_pFDR(input_file,toolbox_path):

    #Add the toolbox to path
    #toolbox_path = "/Users/alblle/allera_version_controlled_code/One_Dim_Mixture_Models/python/code"
    sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 

    #load input conenctivity matrix 
    #input_file="/Users/alblle/Dropbox/POSTDOC/Demetrius/dmn_non_normalized.csv" 
    connectivity_matrix = np.loadtxt(input_file, delimiter=',')#, skiprows=1,skipcolumns=1)

    #get updiagonal terms
    updiag_idx=np.triu_indices_from(connectivity_matrix,k=1)
    orig_data_vector=connectivity_matrix[updiag_idx]
    data_vector=orig_data_vector[orig_data_vector>0.05]
    scaling_factor=np.mean(data_vector)
    data_vector=np.divide(data_vector,scaling_factor)

    #Define options for the mixture model fit
    Inference ='Variational Bayes'  #'Method of moments' OR 'Maximum Likelihood' OR 'Variational Bayes' ML NOT INCLUDED YET
    Number_of_Components=2
    Components_Model=['Gamma','Gamma']#,'-Gamma'] #Each component can be Gauss, Gamma, InvGamma, -Gamma, -InvGamma
    tail=np.percentile(data_vector,99)
    init_params=[1,2,tail,2]#,-5,2]
    maxits=500
    tol=0.00001
    opts={'Inference':Inference,'Number_of_Components':Number_of_Components,'Components_Model':Components_Model,
                                            'init_params':init_params,'maxits':maxits,'tol':tol}
    #Define options for the mixture model fit


    # CALL TO FIT MIXTURE MODEL
    from Mixture_Model_1Dim import Mixture_Model_1Dim     
    Model = Mixture_Model_1Dim(data_vector, opts)
    #print Model['Mixing Prop.']
    # CALL TO FIT MIXTURE MODEL


    if 0:
        # Plot the resulting fit on a histogram of the data
        from alb_MM_functions import gam
        my_range=np.linspace(0.01,np.max(data_vector),10000)
        plt1=np.multiply( Model['Mixing Prop.'][0],gam(my_range,Model['shapes'][0],np.divide(1,Model['rates'][0])))
        plt2=np.multiply( Model['Mixing Prop.'][1],gam(my_range,Model['shapes'][1],np.divide(1,Model['rates'][1])))
        
        import matplotlib.pyplot as plt
        plt.hist(data_vector,bins=50,density=True,alpha=1, color='g')
        plt.plot(my_range,plt1, 'k', linewidth=2)
        plt.plot(my_range,plt2, 'k', linewidth=2)
        plt.plot(my_range,plt1+plt2, 'r', linewidth=2)
        plt.show()
        # Plot the resulting fit on a histogram of the data
        
        
    #Compute local FDR
    p0=Model['Mixing Prop.'][0]
    #f0(x)=gam(x,Model['shapes'][0],np.divide(1,Model['rates'][0])))
    rho=data_vector.shape[0]
    sorted_data_vector=-np.sort(-data_vector)
    all_localFDR=np.ones(rho)
    flag=0
    k=-1
    while flag==0:
        k=k+1
        point=sorted_data_vector[k]
        cdf=scipy.stats.gamma.cdf(point,Model['shapes'][0],0,np.divide(1.,Model['rates'][0]))
        numerator=np.multiply(float(p0),1-cdf)
        denominator=np.divide(float(k+1),float(rho))
        all_localFDR[k]=np.divide(numerator,denominator)
        pFDR=all_localFDR[k]
        if pFDR>0.05:
            threshold=np.multiply(sorted_data_vector[k-1],scaling_factor)
            flag=1
            print threshold
        
    return threshold


