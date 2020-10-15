#from pylab import * 
import warnings
warnings.filterwarnings("ignore")
from SIN_init_VB_MM import SIN_init_VB_MM
import scipy.special as sp  
import numpy as np 
import alb_MM_functions as alb
import copy

def Mix_Mod_VB(x,opts):    
#Vatiational Bayes Mixture model: Gauss/Gamma/Inverse Gamma
#INPUT: x is a data vector;
#       opts 
#OUTPUT: MODEL is a DICTIONARY with the parameters of the fitted model
    
#init 
    Prior, Posterior = SIN_init_VB_MM(x,opts)

    
#gather main options
    tol1=opts['tol']; tol2=opts['tol']; MaxNumIt=opts['maxits'];  K=opts['Number_of_Components']
    MaxNumIt2=1
#gather positive and negative indexes to work around the -Gamma and -InvGamma
    pos=np.where(x>0); neg=np.where(x<0)
    
#---gather priors
    lambda_0=Prior['lambda_0'] 
    # Gauss components precision priors, scale and shape
    b0=Prior['b0']; c0=Prior['c0']
    # Gauss components mean priors
    m_0=Prior['m_0'];tau_0=Prior['tau_0'] 
    # Gamma InvGamma components rate priors, shape and rate
    d_0=Prior['d_0']; e_0=Prior['e_0']  
    # # Gamma InvGamma components  shape priors
    loga_0=Prior['loga_0']; b_0=Prior['b_0']; c_0=Prior['c_0']   

#gather and init necesary posteriors
    mus=Posterior['mus'];tau1s=Posterior['tau1s'];bb=Posterior['b0'];cc=Posterior['c0']
    tau=np.zeros(K);mean_x=np.zeros(K);mean_tau1=np.zeros(K);KLTAU1=np.zeros(K)
    mean_mu12=np.zeros(K); mean_mu1=np.zeros(K); mm=np.zeros(K); KLMU1=np.zeros(K)
    
# --------   Iterate Posterioreriors# 
    ftot=0;
    FEs=np.zeros([1, MaxNumIt+1]); likelihood=np.zeros([1, MaxNumIt+1]); kls=np.zeros([1, MaxNumIt+1]); ents=np.zeros([1, MaxNumIt+1])
    flag=0;it=0; L=np.zeros(K) 
    while flag==0: 
        it=it+1;

        #==================================E_Step================================    

        # compute responsibilitites
        gammas = my_gammas(x,opts,Posterior)   
        Posterior['gammas']=gammas;
        #==================================E_Step================================   

        #==================================M_Step================================
        gamma_sum = gammas.sum(0)    

        #--------------------Update lambda etc.--------------------        
        lambdaq,Posterior['lambda'],lambda_p,ent_gam,KLPI = update_lambda(K,lambda_0,gamma_sum,Posterior,gammas) 
        #--------------------Update lambda etc.--------------------


                #-----------------------Gaussian components-----------------------
        bb, cc, mm, mean_tau1,tau,mean_mu1,mean_mu12,KLTAU1,KLMU1 = update_Gauss_comps(K,opts,Posterior,tau_0,gamma_sum,gammas,x,mus,
                               tau1s,mean_x,bb,cc,KLTAU1,tau,mean_mu1,mean_mu12,KLMU1,b0,c0,mean_tau1,m_0,mm)
        
        Posterior['b0']=copy.deepcopy(bb)
        Posterior['c0']=copy.deepcopy(cc)
        Posterior['m_0']=copy.deepcopy(mm)
        Posterior['tau1s']=copy.deepcopy(mean_tau1)
        Posterior['tau_0']=copy.deepcopy(tau)
        
       

        #--------------------Update non-gaussian components ---------------------   
        Posterior,B,C,logA,mean_x,ee,dd,mean_rates,mean_logrates,mean_scales,mean_logscales, KL_fancy,KLscale, KLrate =update_non_Gauss_comps(x,gammas,K,opts,Posterior,gamma_sum,e_0,d_0,pos,neg,b_0,c_0,loga_0,MaxNumIt2,tol2)


        #===================Compute free Energy================================
        FEs,f_hidd,likelihood,kls,ents,ftot=Compute_free_energy(x,FEs,it,likelihood,kls,ents,Posterior,gammas,opts,K,lambdaq,bb,cc,mean_tau1,mean_mu1,mean_mu12,B,C,
                        mean_rates,mean_logrates,mean_scales,mean_logscales,ent_gam,KLPI,KLMU1,KLTAU1,KLscale,KLrate,KL_fancy)

        #===================Check convergence
        progress=1    
        if it>2:
            progress=np.absolute(np.true_divide(FEs[0][it-1]-FEs[0][it-2], FEs[0][it-1]));
        if (it>MaxNumIt) |( progress < tol1):
            flag=1;


    FEs=FEs[0][0:it];
    likelihood= likelihood[0][0:it];
    kls= kls[0][1:it];
    ents= ents[0][1:it];

#==================================Energy================================    


    Posterior['lambda']=lambdaq;
    Posterior['Final responsibilities']=gammas.T;
    Posterior['Mixing Prop.']=np.true_divide(gammas.sum(0),gammas.sum()) #(sum(gammas,2)/sum(sum(gammas)))#';
    Posterior['mu1']=mean_mu1;
    Posterior['taus1']=mean_tau1;
    Posterior['shapes']=Posterior['shapes'];
    Posterior['ftot']=ftot;
    Posterior['FEs']=FEs;
    Posterior['Likelihood']=likelihood;
    Posterior['its']=it
    for k in range(K):
        if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'):
            Posterior['rates'][k]=mean_rates[k]
        elif (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):
            Posterior['scales']=mean_scales;

    
    return Posterior

def my_gammas(data,opts,Posterior): 
    import scipy.special as sp  
    import numpy as np 
    import alb_MM_functions as alb ; import copy
    pos= np.where(data>0)
    neg= np.where(data<0)
    K=opts['Number_of_Components']
    #expectation on log pi
    dum=Posterior['lambda'] 
    ElogPi=sp.polygamma(0,dum)-sp.polygamma(0,dum.sum());

    #expectations for component 1
    #expectations on mu1 and mu1^2
    dum1=copy.deepcopy(Posterior['m_0']) 
    dum2=copy.deepcopy(Posterior['tau_0']) 
    Emu1=copy.deepcopy(dum1);
    Emu12=np.power(dum1,2)+ np.divide(1,dum2);
    #expectations on tau1 and log tau1
    dum1=copy.deepcopy(Posterior['c0']) #src.Posterior.c0;#shape
    dum2=copy.deepcopy(Posterior['b0']) #src.Posterior.b0;#scale
    Etau1=dum1*dum2;
    Elogtau1=sp.polygamma(0,dum1)+np.log(dum2);         
    #expectations for component 2 and 3
    #expectations on r and log r (r=rate for GGM and scale for GIM)
    dum1=copy.deepcopy(Posterior['d_0']) #ssrc.Posterior.d_0;#shape
    dum2=copy.deepcopy(Posterior['e_0']) #ssrc.Posterior.e_0;#rate
    Er=np.divide(dum1,dum2)
    Elogr=np.zeros(K)
    for k in range(K):
        if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'):
            Elogr[k]=sp.polygamma(0,dum1[k])-np.log(dum2[k]);       
        elif (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):
            Elogr[k]=sp.polygamma(0,dum1[k])-np.log(dum2[k]);
            
    
        

    #expectations on s and ? log(gamma(s))?? (s=shape)
    dum1=copy.deepcopy(Posterior['loga_0']) #src.Posterior.loga_0;
    dum2=copy.deepcopy(Posterior['b_0']) #src.Posterior.b_0;
    dum3=copy.deepcopy(Posterior['c_0']) #src.Posterior.c_0;
    Es=np.zeros(K)
    for k in range(K):
        if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'):
            Es[k]=alb.invpsi( np.divide(  dum1[k]+ np.multiply(dum3[k] , Elogr[k]) , dum2[k]));
        elif (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):
            Es[k]=alb.invpsi( np.divide(-dum1[k]+ np.multiply(dum3[k] , Elogr[k]) , dum2[k]));


    #approximation to E[log(gamma(s))]
    ElogGams=sp.gammaln(Es) + np.divide(1,dum2) +   np.divide(np.multiply(sp.polygamma(2,Es),Es)  , np.multiply(sp.polygamma(1,Es),dum2 )   );   

        
    #compute 'responsibilities'
    resp=np.zeros([data.shape[0],3])

    for k in range(K):
        if opts['Components_Model'][k]=='Gauss':
            dum=np.power(data,2) + Emu12[k]-np.multiply(2*Emu1[k],data)
            dum2=np.divide(Etau1[k],2)
            resp[:,k]= np.exp(ElogPi[k]+np.true_divide(Elogtau1[k],2)-np.true_divide(np.log(2*np.pi),2) - np.multiply( dum, dum2 ))
        elif (opts['Components_Model'][k]=='Gamma'):
            dum=np.exp( ElogPi[k] + np.multiply( Es[k]-1, np.log(data)) +   np.multiply(Es[k],Elogr[k]) -  ElogGams[k] -  np.multiply(Er[k] ,data));            
            #dum=np.exp(  np.matlib.repmat(ElogPi[k],N,1) + np.multiply( np.matlib.repmat(Es[k] -1,N,1), np.transpose(np.log(data))) +  np.matlib.repmat( np.multiply(Es[k],Elogr[k]),N,1) -  np.matlib.repmat( ElogGams[k],N,1) -  np.multiply(np.matlib.repmat( Er[k],N,1) ,np.transpose(data)));
            resp[:,k]=dum;
            resp[neg,k]=0
        elif (opts['Components_Model'][k]=='-Gamma'):
            dum=np.exp( ElogPi[k] + np.multiply( Es[k]-1, np.log(-data)) +   np.multiply(Es[k],Elogr[k]) -  ElogGams[k] -  np.multiply(Er[k] ,-data));            
            resp[:,k]=dum
            resp[pos,k]=0
        elif (opts['Components_Model'][k]=='InvGamma') :
            dum=np.exp( ElogPi[k] - np.multiply( Es[k]+1, np.log(data)) +np.multiply(Es[k],Elogr[k]) -   ElogGams[k] - np.divide(Er[k] ,data));     
            #dum=np.exp(  np.matlib.repmat(ElogPi[k],N,1) - np.multiply( np.matlib.repmat(Es[k] +1,N,1), np.transpose(np.log(data))) +np.matlib.repmat( np.multiply(Es[k],Elogr[k]),N,1) -  np.matlib.repmat( ElogGams[k],N,1) - np.divide(np.matlib.repmat( Er[k],N,1) ,np.transpose(data)));
            resp[:,k]=dum
            resp[neg,k]=0
        elif (opts['Components_Model'][k]=='-InvGamma'):
            dum=np.exp( ElogPi[k] - np.multiply( Es[k]+1, np.log(-data)) +np.multiply(Es[k],Elogr[k]) -   ElogGams[k] - np.divide(Er[k] ,-data));     
            resp[:,k]=dum
            resp[pos,k]=0

    resp= np.asarray(np.divide(resp , np.matrix(np.sum(resp,1)).T))
    
    return resp#[0]


def update_lambda(K,lambda_0,gamma_sum,Posterior,gammas):
            gamma_sum=gamma_sum[range(K)]
            lambdaq = lambda_0+gamma_sum
            Posterior['lambda']=lambdaq   
            # from choud  
            lambda_p=lambda_0*np.ones([1,K]);
            dir1 = sum(sp.gammaln(lambdaq) - sp.gammaln(lambda_p));
            dir2 = sp.gammaln(sum(lambdaq)) - sp.gammaln(sum(lambda_p));
            #Fdir = dir1-dir2;
            ent_gam=-np.nansum(np.multiply(gammas,np.log(gammas))); #ent_gam=-sum(sum(np.multiply(gammas,np.log(gammas))));
    
            #alb-->
            dir3=sum( np.multiply((lambdaq-lambda_0) , (sp.polygamma(0,lambdaq)-sp.polygamma(0,sum(lambdaq)))));
            KLPI=dir2-dir1+dir3
            
            return lambdaq,Posterior['lambda'],lambda_p,ent_gam,KLPI

def update_Gauss_comps(K,opts,Posterior,tau_0,gamma_sum,gammas,x,mus,
                               tau1s,mean_x,bb,cc,KLTAU1,tau,mean_mu1,mean_mu12,KLMU1,b0,c0,mean_tau1,m_0,mm):
            pdf_fact=0.5;  
            for k in range(K):
                if opts['Components_Model'][k]=='Gauss':
            #--------------------Update precisions---------------------                
                    mus[k]= copy.deepcopy(Posterior['mus'][k])#src.mu1;
                    tau1s[k]= copy.deepcopy(Posterior['tau1s'][k])#src.tau1;
                    tmp_tau = tau_0+(tau1s[k]*gamma_sum[k]);
                    mean_xsq = np.multiply(gammas[:,k], np.power(x,2)).sum();
                    mean_x[k] = np.multiply(gammas[:,k], x).sum();
                    #mu_sq = (gamma_sum(1).*(mu1.^2+(1./tau1)))';
                    mu_sq = gamma_sum[k]*(np.power(mus[k],2)+(np.true_divide(1,tmp_tau))); #!!!!!!!
                    data_bit = mean_xsq-(2*mus[k]*mean_x[k])+mu_sq;
                    bb[k] = 1./( 1./b0  + (pdf_fact*data_bit))
                    cc[k] = c0+(pdf_fact*gamma_sum[k]);
                    mean_tau1[k] =bb[k]*cc[k]
                    # contribution to energy using KL of  Gamma, check b=scale and c=shape
                    bp=b0;cp=c0
                    bq=bb[k];cq=cc[k]
                    KLTAU1[k]= (cq*(np.divide(bq,bp)-1)) + ((cq-cp)*(sp.polygamma(0,cq) + np.log(bq))) - sp.gammaln(cq) - ( cq*np.log( bq))   + sp.gammaln(cp) + (cp*np.log( bp));
            
                     #--------------------Update precisions---------------------
    
                    #-----------------------Update means-----------------------
                    tau[k] = tau_0+(mean_tau1[k]*gamma_sum[k]);
                    #mm = 1./tau.*(m_0+mean_tau1.*mean_x'); typo in choud code??
                    mm[k] = np.multiply( 1./tau[k] ,  np.multiply(tau_0,m_0) + np.multiply(mean_tau1[k],mean_x[k]) )                   
                    mean_mu1[k]=copy.deepcopy(mm[k]);
                    mean_mu12[k]=np.power(mm[k],2)+ (1./tau[k])
                   
                    # contribution to energy using KL of Gauss
                    bp=copy.deepcopy(tau_0);mp=copy.deepcopy(m_0);
                    bq=copy.deepcopy(mean_tau1[k]);mq=copy.deepcopy(mean_mu1[k]);
                    KLMU1[k]=0.5 *(  (np.divide(bp,bq)-1)-np.log(np.divide(bp,bq))+ (bp*(np.power(mq-mp,2))) );
                    #-----------------------Update means-----------------------
            return bb, cc, mm, mean_tau1,tau,mean_mu1,mean_mu12,KLTAU1,KLMU1 

def update_non_Gauss_comps(x,gammas,K,opts,Posterior,gamma_sum,e_0,d_0,pos,neg,b_0,c_0,loga_0,MaxNumIt2,tol2):
    subflag=0 #for convergence of Gamma/Inverse Gamma parameters iteration
    its2=0
    mean_x =np.ones(K)
    ee=np.ones(K)
    dd=np.ones(K)
    mean_rates=np.ones(K)
    mean_logrates=np.ones(K)
    mean_scales=np.ones(K)
    mean_logscales=np.ones(K)
    
    while subflag==0:  #for its2=1:subloop
            its2=its2+1
        #--------------------Update rates if GGM or scales if GIM---------------------
            for k in range(K):
                
                if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'):
                    
                    if (opts['Components_Model'][k]=='Gamma'): 
                        mean_x[k] =np.multiply(gammas[:,k],x).sum(0)                              
                    elif (opts['Components_Model'][k]=='-Gamma'): 
                        mean_x[k] =np.multiply(gammas[:,k],-x).sum(0)  
                        
                    ee[k]=e_0[k] + mean_x[k]
                    dd[k]=d_0[k]+np.multiply(Posterior['shapes'][k],gamma_sum[k] )
                    mean_rates[k]=np.divide(dd[k],ee[k])
                    mean_logrates[k]=sp.polygamma(0,dd[k])-np.log(ee[k]) 
                    
                if (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):
                    
                    if (opts['Components_Model'][k]=='InvGamma'):     
                        tmp=np.divide(gammas[:,k],x) 
                        mean_x[k]=np.asarray( np.sum(tmp[pos]))                  #tmp[pos].sum(1)[0])  ;#CHECK IF SUMMING RIGHT DIM

                    elif (opts['Components_Model'][k]=='-InvGamma'):#conj prior on rate of inv gamma is gamma
                        tmp=np.divide(gammas[:,k],-x) 
                        mean_x[k]=np.asarray( np.sum(tmp[neg]))                             
                    #mean_x[k]=np.asarray([  tmp[pos,0].sum(1)[0]  , tmp[neg,1].sum(1)[0] ])  ;#CHECK IF SUMMING RIGHT DIM
                    ee[k]=e_0[k] + mean_x[k]
                    dd[k]= d_0[k] +  np.multiply( Posterior['shapes'][k] , np.asarray(gamma_sum[k]) )
                    mean_scales[k]=np.divide(dd[k],ee[k])
                    mean_logscales[k]=sp.polygamma(0,dd[k])- np.log(ee[k])  
        #--------------------Update rates if GGM or scales if GIM---------------------


        #--------------------Update shapes---------------------
            B=np.ones(K)
            C=np.ones(K)
            logA=np.ones(K)
            for k in range(K):                    
                    if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma') or (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):                            
                        B[k]=  b_0[k]+ gamma_sum[k] 
                        C[k]=  c_0[k]+ gamma_sum[k]
                        if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='InvGamma'):                                                  
                            xresp=np.power(np.transpose(x),gammas[:,k] )
                        else:
                            xresp=np.power(np.transpose(-x),gammas[:,k] )
                            
                        idx= np.where(xresp==0)
                        xresp[idx]=1
                        logA[k]=loga_0[k] +np.log(xresp).sum(0)
                        if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'):
                            Posterior['shapes'][k]=alb.invpsi( np.divide( (logA[k]+ np.multiply(C[k],mean_logrates[k])),B[k]));Posterior['shapes'][k]=np.asarray(Posterior['shapes'][k])
                        else: 
                            Posterior['shapes'][k]=alb.invpsi( np.divide( (-logA[k]+ np.multiply(C[k],mean_logscales[k])),B[k]));Posterior['shapes'][k]=np.asarray(Posterior['shapes'][k])
                            #Posterior['shapes']=alb.invpsi((-logA+ (C .* mean_logscales)) ./ B) ;
        #--------------------Update shapes--------------------- 
                
        #-----check convergence of subloop
            if its2>1: 
                if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'):
                    new=np.hstack([mean_rates, Posterior['shapes']])
                elif (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):
                    new=np.hstack([mean_scales, Posterior['shapes']])
                dude=np.divide(np.abs(old-new),old)    
                mean_rel_change=np.mean(dude[dude>0]);#dummm variable for testing convergence                           
                if (its2>MaxNumIt2) or (mean_rel_change<tol2):
                    subflag=1;
    
            if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'):
                old=np.hstack([mean_rates, Posterior['shapes']]);#dummm variable for testing convergence
            elif (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):    
                old=np.hstack([mean_scales, Posterior['shapes']]);#dummm variable for testing convergence

                    
        #-----check convergence of subloop
        # store for E-step
    Posterior['d_0']=dd
    Posterior['e_0']=ee
    Posterior['loga_0']=logA
    Posterior['b_0']=B
    Posterior['c_0']=C
        
        
    #contribution to energy 
    #f2=@(b,alpha) (b .* psi(1,alpha));# from Laplace approx
    #tmpidx=find(comp(2:3));
    KL_fancy=np.zeros(K)
    KLscale=np.zeros(K)
    KLrate=np.zeros(K)
    for k in range(K):
        if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'):
            #using KL of  Gamma, check b=scale and c=shape?
            bp=np.true_divide(1,e_0[k]);cp=copy.deepcopy(d_0[k]);
            bq=np.true_divide(1,ee[k]);cq=copy.deepcopy(dd[k]);                                   
            KLrates= np.multiply(cq,(np.divide(bq,bp)-1)) + np.multiply((cq-cp),(sp.polygamma(0,cq) + np.log(bq))) - sp.gammaln(cq) - np.multiply( cq,np.log( bq))   + sp.gammaln(cp) + np.multiply(cp,np.log(bp));
            #KLrates=KLrates(tmpidx);
            KLrate[k]=KLrates.sum();
            # using KL of  Gauss and Laplace approx
            logprExpected_rate=sp.polygamma(0,d_0[k])-np.log(e_0[k]);##!!!!!!!get out of loop
            prmean=alb.invpsi(np.divide(loga_0[k]+ (c_0[k] * logprExpected_rate) , b_0[k])); #!!!!!!!get out of loop
            prprecc=alb.f2(b_0[k],prmean);#!!!!!!!get out of loop
            bp=copy.deepcopy(prprecc);
            mp=copy.deepcopy(prmean);#!!!!!!!get out of loop
            bq=alb.f2(B[k],Posterior['shapes'][k]);mq=np.copy(Posterior['shapes'][k]);
            KL_fancy[k]=0.5 *( (np.divide(bp,bq)-1) -np.log(np.divide(bp,bq))+ np.multiply(bp,np.power(mq-mp,2)));
            
               
        elif (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):
            #using KL of  Gamma, check b=scale and c=shape?
            bp=np.true_divide(1,e_0[k]);cp=np.copy(d_0[k]);
            bq=np.true_divide(1,ee[k]);cq=np.copy(dd[k]);    
            KLscales= np.multiply(cq,(np.divide(bq,bp)-1)) + np.multiply((cq-cp),(sp.polygamma(0,cq) + np.log(bq))) - sp.gammaln(cq) - np.multiply( cq,np.log( bq))   + sp.gammaln(cp) + np.multiply(cp,np.log( bp));
            #KLscales=KLscales(tmpidx);
            KLscale[k]=KLscales.sum();  
            
            
            # using KL of  Gauss and Laplace approx
            logprExpected_scale =sp.polygamma(0,d_0[k])-np.log(e_0[k]);##!!!!!!!get out of loop
            prmean=alb.invpsi(np.divide(-loga_0[k]+ (c_0[k] * logprExpected_scale) , b_0[k])); ##!!!!!!!get out of loop
            prprecc=alb.f2(b_0[k],prmean);##!!!!!!!get out of loop
            bp=copy.deepcopy(prprecc);mp=copy.deepcopy(prmean);##!!!!!!!get out of loop
            bq=alb.f2(B[k],Posterior['shapes'][k]);mq=copy.deepcopy(Posterior['shapes'][k]);
            KL_fancy[k]=0.5 *(  (np.divide(bp,bq)-1)-np.log(np.divide(bp,bq))+ np.multiply(bp,np.power(mq-mp,2)  ));
        

    return Posterior, B,C,logA,mean_x,ee,dd,mean_rates,mean_logrates,mean_scales,mean_logscales, KL_fancy,KLscale, KLrate
                        
def Compute_free_energy(x,FEs,it,likelihood,kls,ents,Posterior,gammas,opts,K,lambdaq,bb,cc,mean_tau1,mean_mu1,mean_mu12,B,C,
                        mean_rates,mean_logrates,mean_scales,mean_logscales,ent_gam,KLPI,KLMU1,KLTAU1,KLscale,KLrate,KL_fancy):
#compute tilde pi
    tildepi=np.exp(sp.polygamma(0,lambdaq)-sp.polygamma(0,sum(lambdaq)));
    tildetau1=bb*np.exp(sp.polygamma(0,cc));
    #ggm: tilder=(1./e).*exp(psi(d));
    #gim: tilder=(1./e).*exp(psi(d));
    #FE expected_likelihood contribution (L)
    L=np.zeros(K)
    ElogGams=np.zeros(K)
    #gauss component
    for k in range(K):
        if opts['Components_Model'][k]=='Gauss':
            const_bit=np.multiply(gammas[:,k], (np.log(tildepi[k])+np.true_divide(np.log(tildetau1[k]),2)));
            data_bit= np.multiply(np.transpose(gammas[:,k]),np.transpose(mean_tau1[k]/2.)*(np.power(x,2) + mean_mu12[k] -(2*mean_mu1[k]*x)));
            L[k]=(np.transpose(const_bit)-data_bit).sum()-(np.true_divide( gammas[:,[0]].sum() ,2)*np.log(2*np.pi));#((numel(x)/2)*log(2*pi));
    #non-gauss components
        elif (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma') or (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):                            
            xx=copy.deepcopy(x); #xx;xxx=np.asarray(xxx);#xxx[:,find(xxx==0)]=1;PROBLEM HERE WITH XX AND XXX.... XXX CHANGES VALUES OF XX???????
            xx[np.where(xx==0)]=1;      
            ElogGams[k]=sp.gammaln(Posterior['shapes'][k])+ np.divide(1,B[k]) +   np.divide(  np.multiply(sp.polygamma(2,Posterior['shapes'][k]),Posterior['shapes'][k]) , np.multiply(sp.polygamma(1,Posterior['shapes'][k]),B[k] )   );
            #ElogGams=gammaln(Posterior['shapes'])+ (1./B)+   ((psi(2,Posterior['shapes']).*Posterior['shapes'])./ (psi(1,Posterior['shapes']).*B )   ); SAME FOR GGM AND GIM ??
            if (opts['Components_Model'][k]=='Gamma') or (opts['Components_Model'][k]=='-Gamma'): 
                if (opts['Components_Model'][k]=='Gamma'): 
                    arrg=np.multiply(  (Posterior['shapes'][k]-1) , np.log(xx)) - np.multiply( mean_rates[k],x) #or xx?
                elif (opts['Components_Model'][k]=='-Gamma'):    
                    arrg=np.multiply(  (Posterior['shapes'][k]-1) , np.log(-xx)) - np.multiply( mean_rates[k],-x); #or xx?

                data_bit=np.multiply(gammas[:,k], arrg)            
                const_bit=(np.log(tildepi[k])+ np.multiply(Posterior['shapes'][k],mean_logrates[k])-ElogGams[k])#';
                const_bit= np.multiply(gammas[:,k], const_bit); 

            elif (opts['Components_Model'][k]=='InvGamma') or (opts['Components_Model'][k]=='-InvGamma'):
                if (opts['Components_Model'][k]=='InvGamma'): 
                    arrg=np.multiply( (Posterior['shapes'][k]+1) , -np.transpose(np.log(xx))) - np.divide(  (mean_scales[k]), x)
                elif (opts['Components_Model'][k]=='-InvGamma'): 
                    arrg=np.multiply( (Posterior['shapes'][k]+1) , -np.transpose(np.log(-xx))) - np.divide(  (mean_scales[k]), -x)

                data_bit=np.multiply(gammas[:,k], arrg)
                const_bit=(np.log(tildepi[k])+ np.multiply(Posterior['shapes'][k],mean_logscales[k])-ElogGams[k])#';
                const_bit= np.multiply(gammas[:,k],  const_bit);
                
            data_bit[np.isnan(data_bit)]=0    
            L[k]=np.sum(const_bit+data_bit)

   
    LIK=L.sum()
    KLs=np.sum(KLPI+KLMU1+KLTAU1+KLscale+KLrate+KL_fancy) 



    f_hidd = ent_gam;#-numel(x)/2*log(2*pi);
    ftot=LIK+ent_gam-KLs;
    FEs[0][it-1]=ftot;
    likelihood[0][it-1]=LIK;
    kls[0][it-1]=-KLs;
    ents[0][it-1]=f_hidd; 
    return FEs,f_hidd,likelihood,kls,ents,ftot                        
