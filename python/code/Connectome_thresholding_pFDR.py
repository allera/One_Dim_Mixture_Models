import numpy as np
import sys
import os
import scipy
# Plot the resulting fit on a histogram of the data
from alb_MM_functions import invgam
from alb_MM_functions import gam
from scipy.stats import norm

def GaussGammas_Connectome_thresholding_pFDR(input_file,toolbox_path):

    #Add the toolbox to path
    sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 
    from Mixture_Model_1Dim import Mixture_Model_1Dim     

    #load input conenctivity matrix 
    #connectivity_matrix = np.loadtxt(input_file, delimiter=',')#, skiprows=1,skipcolumns=1)
    connectivity_matrix =np.genfromtxt(input_file, delimiter=',')

    #get updiagonal terms
    updiag_idx=np.triu_indices_from(connectivity_matrix,k=1)
    orig_data_vector=connectivity_matrix[updiag_idx]
    orig_data_vector=orig_data_vector[~np.isnan(orig_data_vector)] #data_vector=orig_data_vector[orig_data_vector>0.05]
    
    #demean and divide for std to allow easy initialization
    mean_factor=np.mean(orig_data_vector)
    scaling_factor=np.std(orig_data_vector)
    data_vector=np.divide(orig_data_vector - mean_factor,scaling_factor)

    #Define options for the mixture model fit
    Inference ='Variational Bayes'#'Method of moments'#'Variational Bayes' #'Variational Bayes'  #'Method of moments' OR 'Maximum Likelihood' OR 'Variational Bayes' ML NOT INCLUDED YET
    Number_of_Components=3
    Components_Model=['Gauss','InvGamma','-InvGamma']#,'-Gamma'] #Each component can be Gauss, Gamma, InvGamma, -Gamma, -InvGamma
    maxits=500
    tol=0.00001    
    init_params=[1,2,5,2,-5,2]    
    #tail=np.percentile(data_vector,percentiles[percentile_idx])
    #init_params=[1,2,np.percentile(data_vector,99),2,np.percentile(data_vector,1),2]
    opts={'Inference':Inference,'Number_of_Components':Number_of_Components,'Components_Model':Components_Model,
                                            'init_params':init_params,'maxits':maxits,'tol':tol}
    # CALL TO FIT MIXTURE MODEL
    Model = Mixture_Model_1Dim(data_vector, opts)
    #if Model['Mixing Prop.'][0]<.95:
    #good_model=1



    # Visualizar fit
    visualize_model_fit=1
    
    if visualize_model_fit==1:
        
        
        my_range=np.linspace(-10,10,10000)

        plt0=np.multiply( Model['Mixing Prop.'][0],norm.pdf(my_range,Model['mu1'][0],np.sqrt(np.divide(1,Model['taus1'][0]))  ) )
        #plt0=np.multiply( Model['Mixing Prop.'][0],norm.pdf(my_range,Model['mu1'][0],np.sqrt(Model['taus1'][0])  ) )
        #plt0=np.multiply( Model['Mixing Prop.'][0],norm.pdf(my_range,Model['mu1'][0],Model['taus1'][0])  ) 


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


        import matplotlib.pyplot as plt
        plt.hist(data_vector,bins=50,density=True,alpha=1, color='g')
        plt.plot(my_range,plt0, 'k', linewidth=2)
        plt.plot(my_range,plt1, 'k', linewidth=2)
        plt.plot(my_range,plt2, 'k', linewidth=2)
        plt.plot(my_range,plt0+plt1+plt2, 'r', linewidth=2)
        plt.show()
        # Plot the resulting fit on a histogram of the data
        
        
    #Compute local FDR at positive and negative tail 
    #f0(x)=gam(x,Model['shapes'][0],np.divide(1,Model['rates'][0])))
    p0=Model['Mixing Prop.'][0]
    rho=data_vector.shape[0]
    
    #FDR at positive side
    sorted_data_vector=-np.sort(-data_vector)
    all_localFDR=np.ones(rho)
    flag=0
    k=-1
    while flag==0:
        k=k+1
        point=sorted_data_vector[k]
        cdf=norm.cdf(point,Model['mu1'][0],np.sqrt(np.divide(1,Model['taus1'][0])))
        numerator=np.multiply(float(p0),1-cdf)
        denominator=np.divide(float(k+1),float(rho))
        all_localFDR[k]=np.divide(numerator,denominator)
        pFDR=all_localFDR[k]
        if pFDR>0.05:
            if k==0:
                threshold1=sorted_data_vector[k]
            else:
                threshold1=sorted_data_vector[k-1]; # np.multiply(sorted_data_vector[k-1],scaling_factor)
                
            flag=1
            
            print threshold1
            
    #FDR at negative side
    sorted_data_vector=-np.sort(data_vector)
    all_localFDR=np.ones(rho)
    flag=0
    k=-1
    while flag==0:
        k=k+1
        point=sorted_data_vector[k]
        cdf=norm.cdf(-point,Model['mu1'][0],np.sqrt(np.divide(1,Model['taus1'][0])))
        numerator=np.multiply(float(p0),1-cdf)
        denominator=np.divide(float(k+1),float(rho))
        all_localFDR[k]=np.divide(numerator,denominator)
        pFDR=all_localFDR[k]
        if pFDR>0.05:
            if k==0:
                threshold2=-sorted_data_vector[k]
            else:
                threshold2=-sorted_data_vector[k-1]; # np.multiply(sorted_data_vector[k-1],scaling_factor)
                
            flag=1
            
            
            
    #Rescale the thresholds using the data mean and std    
    threshold1= np.multiply(threshold1,scaling_factor) + mean_factor
    threshold2= np.multiply(threshold2,scaling_factor) + mean_factor
    print threshold1
    print threshold2

 
    return threshold1, threshold2, Model







def GammaGamma_Connectome_thresholding_pFDR(input_file,toolbox_path):

    #Add the toolbox to path
    #toolbox_path = "/Users/alblle/allera_version_controlled_code/One_Dim_Mixture_Models/python/code"
    sys.path.append(os.path.join(os.path.abspath(toolbox_path))) 
    from Mixture_Model_1Dim import Mixture_Model_1Dim     

    #load input conenctivity matrix 
    #input_file="/Users/alblle/Dropbox/POSTDOC/Demetrius/dmn_non_normalized.csv" 
    #connectivity_matrix = np.loadtxt(input_file, delimiter=',')#, skiprows=1,skipcolumns=1)
    connectivity_matrix =np.genfromtxt(input_file, delimiter=',')


    #get updiagonal terms
    updiag_idx=np.triu_indices_from(connectivity_matrix,k=1)
    orig_data_vector=connectivity_matrix[updiag_idx]
    orig_data_vector=orig_data_vector[~np.isnan(orig_data_vector)]
    data_vector=orig_data_vector[orig_data_vector>0.05]
    scaling_factor=np.mean(data_vector)
    data_vector=np.divide(data_vector,scaling_factor)

    #Define options for the mixture model fit
    Inference ='Variational Bayes'  #'Method of moments' OR 'Maximum Likelihood' OR 'Variational Bayes' ML NOT INCLUDED YET
    Number_of_Components=2
    Components_Model=['Gamma','InvGamma']#,'-Gamma'] #Each component can be Gauss, Gamma, InvGamma, -Gamma, -InvGamma
    maxits=500
    tol=0.00001
    good_model=0
    percentiles=np.array([99,98.5,98,97.5,97,96.5,96,95.5,95])
    percentile_idx=-1
    while good_model==0:
        percentile_idx=percentile_idx+1
        tail=np.percentile(data_vector,percentiles[percentile_idx])
        init_params=[1,2,tail,2]#,-5,2]
        opts={'Inference':Inference,'Number_of_Components':Number_of_Components,'Components_Model':Components_Model,
                                            'init_params':init_params,'maxits':maxits,'tol':tol}
    #Define options for the mixture model fit
        # CALL TO FIT MIXTURE MODEL
        Model = Mixture_Model_1Dim(data_vector, opts)
        #if Model['Mixing Prop.'][0]<.95:
        good_model=1
        # CALL TO FIT MIXTURE MODEL


    if 1:
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
        
    return threshold, Model


