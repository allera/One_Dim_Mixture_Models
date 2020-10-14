
import alb_MM_functions as alb
import math
import numpy as np
import scipy.stats
from scipy import special
import warnings
warnings.filterwarnings("ignore")
import copy

def Mix_Mod_ML(x, opts={'Number_of_Components':3,'Components_Model':['Gauss','Gamma','-Gamma'],
                    'init_params':[0,1,3,1,-3,1],'maxits':np.int(100),'tol':0.00001,'init_pi':np.true_divide(np.ones(3),3)}): 
    
    
    param1,param2,maxiters,tol,K,tmp_PI,MM_Exp_lik= init_ML(x,opts)
    
    #indexes of samples to assign 0 prob wrt each positive definite distr.    
    xneg=np.argwhere(x<pow(10,-14))[:,0]
    xpos=np.argwhere(x>-pow(10,-14))[:,0]
    
    #ITERATE
    flag=0
    it=0
    Exp_lik=np.zeros(opts['maxits']+1)
    while flag==0:
        # E-step    
        PS,resp,tmp_PI,N,Exp_lik[it] = ML_E_step(x,K,opts,param1,param2,tmp_PI,xpos,xneg)
        #M-step
        param1, param2 = ML_M_step(x,K,param1,param2,resp,N,opts,xpos,xneg)
        #convergence criterium
        if it>0:
            if (abs((Exp_lik[it]-Exp_lik[it-1])/Exp_lik[it-1] )< tol) | (it > maxiters-1):
                flag=1
        it=it+1
        
    #gather output
    tmp_mu=np.zeros(K) #it will remain zero for non-gauss
    tmp_v = np.zeros(K)  

    tmp_a=np.zeros(K) #it will remain zero for non-gamma or inv gamma distributions
    tmp_b=np.zeros(K) #it will remain zero for non-gamma or inv gamma distributions  
    tmp_c=np.zeros(K)
    for k in range(K):
        if opts['Components_Model'][k]=='Gauss': 
            tmp_mu[k] = param1[k]
            tmp_v[k] = param2[k]    
        elif ((opts['Components_Model'][k]=='Gamma') | (opts['Components_Model'][k]=='-Gamma')): 
            tmp_a[k] =  param1[k]
            tmp_b[k] =  param2[k]    
        elif ( (opts['Components_Model'][k]=='InvGamma') | (opts['Components_Model'][k]=='-InvGamma') ): 
            tmp_a[k] =  param1[k]
            tmp_c[k] =  param2[k]
        elif opts['Components_Model'][k]=='Beta': 
            tmp_a[k] =param1[k] #alb.a_beta_distr(param1[k],param2[k])
            tmp_c[k] =param2[k] #alb.b_beta_distr(param1[k],param2[k])
            
            
    output_dict={'means':tmp_mu,'mu1':tmp_mu,'variances':tmp_v,'taus1':np.divide(1,tmp_v),'Mixing Prop.':np.asarray(tmp_PI)[0],
                 'Likelihood':Exp_lik[0:it],'its':it,'Final responsibilities':resp,
                 'opts':opts,'shapes':tmp_a,'scales':tmp_c,'rates':tmp_b}

    return output_dict 

def init_ML(x,opts):
    
    tol=opts['tol']
    maxiters=opts['maxits']
    K=opts['Number_of_Components']
    #Exp_lik=np.zeros(maxiters+1)

    opts_MM=copy.deepcopy(opts)
    opts_MM['maxits']=np.int(1)
    Model = alb.Mix_Mod_MethodOfMoments(x, opts_MM)
    Exp_lik=Model['Likelihood']
    
    tmp_mu=Model['mu1']
    tmp_v=Model['variances']
    tmp_PI=Model['Mixing Prop.']
    param1=np.zeros(K)
    param2=np.zeros(K)
    for k in range(K):
        if opts['Components_Model'][k]=='Gauss':
            param1[k]=tmp_mu[k]
            param2[k]=tmp_v[k]
        elif opts['Components_Model'][k]=='Gamma': 
            param1[k] =alb.alphaGm(tmp_mu[k],tmp_v[k])
            param2[k] =np.divide(1.,alb.betaGm(tmp_mu[k],tmp_v[k]))    
        elif opts['Components_Model'][k]=='-Gamma': 
            param1[k] =alb.alphaGm(-1*tmp_mu[k],tmp_v[k])
            param2[k] =np.divide(1.,alb.betaGm(-1*tmp_mu[k],tmp_v[k]))     
        elif opts['Components_Model'][k]=='InvGamma': 
            param1[k] =alb.alphaIG(tmp_mu[k],tmp_v[k])
            param2[k] =alb.betaIG(tmp_mu[k],tmp_v[k])
        elif opts['Components_Model'][k]=='-InvGamma': 
            param1[k] =alb.alphaIG(-1*tmp_mu[k],tmp_v[k])
            param2[k] =alb.betaIG(-1*tmp_mu[k],tmp_v[k])
        elif opts['Components_Model'][k]=='Beta': 
            param1[k] =alb.a_beta_distr(tmp_mu[k],tmp_v[k])
            param2[k] =alb.b_beta_distr(tmp_mu[k],tmp_v[k])
    
    
    
    return param1,param2,maxiters,tol,K,tmp_PI,Exp_lik


def ML_E_step(x,K,opts,param1,param2,tmp_PI,xpos,xneg):
    PS=np.zeros([K,x.shape[0]])
    D=np.zeros([K,x.shape[0]]) # storages probability of samples wrt distributions
    
    tmp_a=np.zeros(K) #it will remain zero for non-gamma or inv gamma distributions
    tmp_b=np.zeros(K) #it will remain zero for non-gamma or inv gamma distributions   
    for k in range(K):
        if opts['Components_Model'][k]=='Gauss':        
            Nobj=scipy.stats.norm(param1[k],np.power(param2[k],0.5));
            PS[k,:]=Nobj.pdf(x);            
        elif opts['Components_Model'][k]=='Gamma':     
            PS[k,:]=alb.gam_self(x,param1[k], 1/param2[k]);
            PS[k,xneg]=0
        elif opts['Components_Model'][k]=='-Gamma':     
            PS[k,:]=alb.gam_self(-1*x,param1[k], 1/param2[k]);
            PS[k,xpos]=0
        elif opts['Components_Model'][k]=='InvGamma': 
            PS[k,:]=alb.invgam(x,param1[k], param2[k])
            PS[k,xneg]=0
        elif opts['Components_Model'][k]=='-InvGamma': 
            PS[k,:]=alb.invgam(-1*x,param1[k], param2[k])
            PS[k,xpos]=0
        elif opts['Components_Model'][k]=='Beta': 
            PS[k,:]=scipy.stats.beta.pdf(x,param1[k], param2[k])
            
            
    PS[np.isnan(PS)] = 0; PS[np.isinf(PS)] = 0
    D=np.multiply(PS,np.matrix(tmp_PI).T) 
    resp= np.divide(D,np.matrix(np.sum(D,0) ))
    N=np.sum(resp,1)   
    tmp_PI=np.divide(N,np.sum(resp)).T
    if 0:
        dum=np.add(np.log(PS),np.log(tmp_PI).T) 
        dum[np.isinf(dum)]=0
        dum[np.isinf(dum)]=0
        Exp_lik=np.sum(np.multiply(resp,dum))
    else:
        dum=np.multiply(tmp_PI.T,PS) #add(np.log(PS),np.log(tmp_PI).T) 
        dum[np.isinf(dum)]=1
        dum[np.isinf(dum)]=1
        dum[dum==0]=1

        Exp_lik=np.sum(np.log(dum))
    
    return PS,resp,tmp_PI,N,Exp_lik

def ML_M_step(x,K,param1,param2,resp,N,opts,xpos,xneg):
    for k in range(K):
        if opts['Components_Model'][k]=='Gauss':        
            param1[k] =np.sum(np.multiply(resp[k,:],x))/N[k]
            param2[k]=np.sum(np.multiply(resp[k,:],np.square(x-param1[k])))/N[k]
        
        if ( (opts['Components_Model'][k]=='Gamma') | (opts['Components_Model'][k]=='-Gamma')): 
            #%ML parameters estimation similar to minka MODEL 2,fastest
            if opts['Components_Model'][k]=='Gamma':
                xx=x[xpos]
                m=np.sum(np.multiply(resp[k,xpos],xx))/N[k] 
                v=np.sum(np.multiply(resp[k,xpos],np.square(xx-m)))/N[k]  
                ml=np.sum(np.multiply(np.log(xx),resp[k,xpos]))/N[k] 

            else:
                xx=-x[xneg]
                m=np.sum(np.multiply(resp[k,xneg],xx))/N[k] 
                v=np.sum(np.multiply(resp[k,xneg],np.square(xx-m)))/N[k]  
                ml=np.sum(np.multiply(np.log(xx),resp[k,xneg]))/N[k] 

            lm=np.log(m)
            aa=param1[k]
            start_aa=np.copy(aa)
           
            flag=0;
            its2=0
            while flag==0: 
                its2=its2+1
                dum=(1/aa) + ( (ml - lm + np.log(aa) -special.polygamma(0,aa)) / (np.power(aa,2)* ((1/aa)-special.polygamma(1,aa))) )
                old_aa=np.copy(aa)
                aa=1/dum;
                df=abs(aa-old_aa)
                if ((df < opts['tol']) | (its2 > 20)):
                    flag=1

            param1[k]=aa 
            param2[k]=1/(m/aa) # the actual update in the paper is for the scale so needs invertion to return rate...

        
            
        elif ((opts['Components_Model'][k]=='InvGamma') | (opts['Components_Model'][k]=='-InvGamma')) : 
        #ML parameters estimation using our derivations inspired by gamma minka MODEL 2, using approx to
        # likelihood p(a)=c0 + c1a+c2 log a
            if opts['Components_Model'][k]=='InvGamma':
                xx=x[xpos]
                m=np.sum(np.multiply(resp[k,xpos],xx))/N[k] 
                v=np.sum(np.multiply(resp[k,xpos],np.square(xx-m)))/N[k] 
                ml=np.sum(np.multiply(np.log(xx),resp[k,xpos]))/N[k] 
                Lis=np.log(np.sum(np.multiply(resp[k,xpos],1./xx))) 

            else:
                xx=-x[xneg]
                m=np.sum(np.multiply(resp[k,xneg],xx))/N[k] 
                v=np.sum(np.multiply(resp[k,xneg],np.square(xx-m)))/N[k] 
                ml=np.sum(np.multiply(np.log(xx),resp[k,xneg]))/N[k] 
                #Lis=np.log(np.sum(1./xx)) 
                Lis=np.log(np.sum(np.multiply(resp[k,xneg],1./xx))) 
                
            ns=N[k] 
            aa=param1[k]
            init_aa=np.copy(aa)
            p2=param2[k]
           
            flag=0; 
            its2=0
            while flag==0: 
                its2=its2+1
                dum=(1/aa) + (  ( (-ml-special.polygamma(0,aa)+ np.log(ns*aa) - Lis ))  /   (np.power(aa,2)*( ((1/aa)-special.polygamma(1,aa))))  )
                old_aa=np.copy(aa)
                aa=1/dum;
                df=abs(aa-old_aa)
                if ((df < opts['tol']) | (its2 > 20)):
                    flag=1

            param1[k]=aa 
            
            if opts['Components_Model'][k]=='InvGamma':
                param2[k]=(ns * aa) / np.sum(np.multiply(resp[k,xpos],1./xx)) 
            else:
                param2[k]=(ns * aa) / np.sum(np.multiply(resp[k,xneg],1./xx)) 


        elif opts['Components_Model'][k]=='Beta': 
            
            p1=copy.deepcopy(param1[k])
            p2=copy.deepcopy(param2[k])
            
            
            
            flag=0
            its2=-1
            while flag==0:
                its2=its2+1
                aa=special.polygamma(0,p1+p2)
                bb=special.polygamma(0,p1)
                cc=special.polygamma(0,p2)
                ml=np.sum(np.multiply(np.log(x),resp[k,:]))/N[k] 
                ml2=np.sum(np.multiply(np.log(1-x),resp[k,:]))/N[k] 
                
                g=np.zeros([1,2])
                g1=aa-bb+ml
                g2=aa-cc+ml2
                g[0,0]=g1
                g[0,1]=g2
                
                G=np.zeros([2,2])
                g12=special.polygamma(1,p1+p2)
                g11=g12-special.polygamma(1,p1)
                g22=g12-special.polygamma(1,p2)
                G[0,0]=g11
                G[0,1]=g12
                G[1,0]=g12
                G[1,1]=g22
                
                
                Y0=np.array([p1,p2])
                Y1=(Y0- np.squeeze((np.dot(np.linalg.inv(G),g.T))))     #Y0 -  (np.dot(np.linalg.inv(G),g)) 
                                
                change=np.linalg.norm(Y0-Y1)
                #print(change)

                p1=Y1[0]
                p2=Y1[1]
                
                if (change < opts['tol']) | (its2>20):
                    flag=1
                
            
            param1[k]= p1
            param2[k]= p2
            #param1[k] =np.sum(np.multiply(resp[k,:],x))/N[k]
            #param2[k]=np.sum(np.multiply(resp[k,:],np.square(x-param1[k])))/N[k]


    return param1,param2


#iNVERSE GAMMA PDF,InvG=invgam(GS,2,3): IN MATLAB invgam = @(x,a,b) b^a/gamma(a).*(1./x).^(a+1).*exp(-b./x);
def invgam(x,aa,bb):
    #out=np.multiply(np.ones(x.shape[0]) *np.divide(np.power(bb,aa),math.gamma(aa)), np.multiply(np.power(np.divide(np.ones(size(x)),x),aa+1) ,np.exp(np.divide(np.ones(size(x))*-bb,x))));
    out=np.multiply(np.ones(x.shape[0]) *np.divide(np.power(bb,aa),math.gamma(aa)), np.multiply(np.power(np.divide(np.ones(x.shape[0]),x),aa+1) ,np.exp(np.divide(np.ones(x.shape[0])*-bb,x))));
    return out;

def gam_old(x,aa,bb):
    #out=np.multiply(np.multiply(np.true_divide(1,np.multiply(np.power(bb,aa),math.gamma(aa))), np.power(x,aa-1)),np.exp(np.divide(np.ones(size(x))*-x,bb)));
    out=np.multiply(np.multiply(np.true_divide(1,np.multiply(np.power(bb,aa),math.gamma(aa))), np.power(x,aa-1)),np.exp(np.divide(np.ones(x.shape[0])*-x,bb)));

    return out;

def gam(x,aa,bb):
    import scipy.stats
    out=scipy.stats.gamma.pdf(x,aa,0,bb)
    return out; 
 
#define functions to translate parameters of InvGamma dist from alpha,beta  <-----> mu, var
def muIG(alpha,beta):
    return np.true_divide(beta,(alpha-1)); #alpha>1

def varIG(alpha,beta):
    return np.true_divide( np.power(beta,2) , ((alpha-2) * (alpha-1)^2 ) );

def alphaIG(mu,var):
    aa=np.true_divide(np.power(mu, 2),var)+2;
    #added hack to avoid overflow that almost never happen!!
    while aa > 160: # above 160 math.gamma(aa) is just numerical error and above 180 overflow 
        var=var+0.001
        aa=np.true_divide(np.power(mu, 2),var)+2;
    
    return aa

def betaIG(mu,var): 
    bb=np.multiply(mu,np.true_divide(np.power(mu, 2),var)+1);
    
    #loop to make sure we make the same alpha as in previous function in case it needed to be hacked to avoid overflow
    aa=np.true_divide(np.power(mu, 2),var)+2;
    #added hack to avoid overflow that almost never happen!!
    while aa > 160: # above 160 math.gamma(aa) is just numerical error and above 180 overflow 
        var=var+0.001
        aa=np.true_divide(np.power(mu, 2),var)+2;
        bb=np.multiply(mu,np.true_divide(np.power(mu, 2),var)+1);

    return bb

def alphaIG_raw(mu,var):
    aa=np.true_divide(np.power(mu, 2),var)+2;    
    return aa

def betaIG_raw(mu,var): 
    bb=np.multiply(mu,np.true_divide(np.power(mu, 2),var)+1);
    return bb


def alphaGm(mu,var):
    aa=np.true_divide(np.power(mu, 2),var);
    return aa

def betaGm(mu,var):
    aa=np.true_divide(var,mu);
    return aa

#m=mean(x);
#s=std(x)^2;
#a_MM=@(m,s)m*((m*(1-m)/s) -1);
#b_MM=@(m,s)(1-m)*((m*(1-m)/s)-1) ;

def a_beta_distr(mm,ss):
    a=np.multiply(mm,( np.divide(np.multiply(mm,(1-mm)),ss) -1))
    return a

def b_beta_distr(mm,ss):
    b=np.multiply(1-mm,( np.divide(np.multiply(mm,(1-mm)),ss) -1))
    return b

def   invpsi(X): 
# Inverse digamma (psi) function.  The digamma function is the derivative of the log gamma function.  
#This calculates the value Y > 0 for a value X such that digamma(Y) = X.
#This algorithm is from Paul Fackler: http://www4.ncsu.edu/~pfackler/
	import scipy.special as sp  
	import numpy as np 
	L = 1;
	Y = np.exp(X);
	while L > 10e-9:
		Y = Y + L*np.sign(X-sp.polygamma(0,Y));
		L = np.true_divide(L , 2);
	return Y

def f2(b,alpha):
	import scipy.special as sp  
	out= np.multiply(b , sp.polygamma(1,alpha));# Necesaty for Laplace approx in VB
	return out

