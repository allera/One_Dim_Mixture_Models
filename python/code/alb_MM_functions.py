
import alb_MM_functions as alb
import math
from pylab import find, size  
import numpy as np;
import scipy.stats
import warnings
warnings.filterwarnings("ignore")

def Mix_Mod_MethodOfMoments(x, opts={'Number_of_Components':3,'Components_Model':['Gauss','Gamma','-Gamma'],
                    'init_params':[0,1,3,1,-3,1],'maxits':np.int(100),'tol':0.00001}): 
    
    tmp_mu,tmp_v,maxiters,tol,K,tmp_PI,Exp_lik = init_MM(x,opts)
    #indexes of samples to assign 0 prob wrt each positive definite distr.
    xneg=find(x<pow(10,-14))
    xpos=find(x>-pow(10,-14))
    #ITERATE
    flag=0
    it=0
    while flag==0:
        # E-step    
        PS,resp,tmp_PI,N,Exp_lik[it] = MM_E_step(x,K,opts,tmp_mu,tmp_v,tmp_PI,xpos,xneg)
        #M-step
        tmp_mu, tm_v= MM_M_step(x,K,tmp_mu,tmp_v,resp,N)
        #convergence criterium
        if it>0:
            if (abs((Exp_lik[it]-Exp_lik[it-1])/Exp_lik[it-1] )< tol) | (it > maxiters-1):
                flag=1
        it=it+1
    #gather output
    output_dict={'means':tmp_mu,'variances':tmp_v,'Mixing Prop.':tmp_PI,
                 'Likelihood':Exp_lik[0:it],'its':it,'Final responsibilities':resp,
                 'opts':opts}

    return output_dict 

def init_MM(x,opts):
    maxiters=opts['maxits']
    tol=opts['tol']
    K=opts['Number_of_Components']
    all_params=opts['init_params']
    #init means and variances
    tmp_mu=np.zeros(K)
    tmp_v=np.zeros(K)
    for k in range(K):
        tmp_mu[k]=all_params[np.int(2*k)]
        tmp_v[k]=all_params[np.int((2*k)+1)]
    tmp_PI=np.true_divide(np.ones(K),K)        
    Exp_lik=np.zeros(maxiters+1)
    
    return tmp_mu,tmp_v,maxiters,tol,K,tmp_PI,Exp_lik


def MM_E_step(x,K,opts,tmp_mu,tmp_v,tmp_PI,xpos,xneg):
    PS=np.zeros([K,size(x)])
    D=np.zeros([K,size(x)]) # storages probability of samples wrt distributions
    tmp_a=np.zeros(K) #it will remain zero for non-gamma or inv gamma distributions
    tmp_b=np.zeros(K) #it will remain zero for non-gamma or inv gamma distributions   
    for k in range(K):
        if opts['Components_Model'][k]=='Gauss':        
            Nobj=scipy.stats.norm(tmp_mu[k],np.power(tmp_v[k],0.5));
            PS[k,:]=Nobj.pdf(x);            
        if opts['Components_Model'][k]=='Gamma': 
            tmp_a[k] =alb.alphaGm(tmp_mu[k],tmp_v[k])
            tmp_b[k] =alb.betaGm(tmp_mu[k],tmp_v[k])    
            PS[k,:]=alb.gam(x,tmp_a[k], tmp_b[k]);
            PS[k,xneg]=0
        if opts['Components_Model'][k]=='-Gamma': 
            tmp_a[k] =alb.alphaGm(-1*tmp_mu[k],tmp_v[k])
            tmp_b[k] =alb.betaGm(-1*tmp_mu[k],tmp_v[k])     
            PS[k,:]=alb.gam(-1*x,tmp_a[k], tmp_b[k]);
            PS[k,xpos]=0
        if opts['Components_Model'][k]=='InvGamma': 
            tmp_a[k] =alb.alphaIG(tmp_mu[k],tmp_v[k])
            tmp_b[k] =alb.betaIG(tmp_mu[k],tmp_v[k])
            PS[k,:]=alb.invgam(x,tmp_a[k], tmp_b[k])
            PS[k,xneg]=0
        if opts['Components_Model'][k]=='-InvGamma': 
            tmp_a[k] =alb.alphaIG(-1*tmp_mu[k],tmp_v[k])
            tmp_b[k] =alb.betaIG(-1*tmp_mu[k],tmp_v[k])
            PS[k,:]=alb.invgam(-1*x,tmp_a[k], tmp_b[k])
            PS[k,xpos]=0
            
    PS[np.isnan(PS)] = 0; PS[np.isinf(PS)] = 0
    D=np.multiply(PS,np.matrix(tmp_PI).T) 
    resp= np.divide(D,np.matrix(np.sum(D,0) ))
    N=np.sum(resp,1)   
    tmp_PI=np.divide(N,np.sum(resp)).T
    dum=np.add(np.log(PS),np.log(tmp_PI).T) 
    dum[np.isinf(dum)]=0
    dum[np.isinf(dum)]=0
    Exp_lik=np.sum(np.multiply(resp,dum))
    
    return PS,resp,tmp_PI,N,Exp_lik

def MM_M_step(x,K,tmp_mu,tmp_v,resp,N):
    for k in range(K):
        tmp_mu[k] =np.sum(np.multiply(resp[k,:],x))/N[k]
        tmp_v[k]=np.sum(np.multiply(resp[k,:],np.square(x-tmp_mu[k])))/N[k]
    return tmp_mu,tmp_v

#RANDOM GAUSS GENERATOR: you can call alb_rndn function as > GS = alb_rndn(100,0,1);
def rndn(n,mu,sigma):
    Gauss_samples = np.random.normal(mu, sigma, n);
    GS=Gauss_samples;
    return GS;
    
#RANDOM GAMMA GENERATOR: you can call alb_rnd_gamma function as gms = alb_rnd_gamma(3,5,10);
def rnd_gamma(alpha,beta,n):
    Gamma_samples = np.random.gamma(alpha, beta,n);
    gms=Gamma_samples;
    return gms;

#def normpdf(x,m,v):
#    out=np.multiply(  pow( 2*math.pi*(pow(v,2)),-0.5 ) ,  exp(np.divide(pow(x-(m*np.ones(size(x))),2),(-2*v)*np.ones(size(x)))); 
#    return out;

#iNVERSE GAMMA PDF,InvG=invgam(GS,2,3): IN MATLAB invgam = @(x,a,b) b^a/gamma(a).*(1./x).^(a+1).*exp(-b./x);
def invgam(x,aa,bb):
    out=np.multiply(np.ones(x.shape[0]) *np.divide(np.power(bb,aa),math.gamma(aa)), np.multiply(np.power(np.divide(np.ones(size(x)),x),aa+1) ,np.exp(np.divide(np.ones(size(x))*-bb,x))));
    return out;

def gam_old(x,aa,bb):
    out=np.multiply(np.multiply(np.true_divide(1,np.multiply(np.power(bb,aa),math.gamma(aa))), np.power(x,aa-1)),np.exp(np.divide(np.ones(size(x))*-x,bb)));
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
    return aa

def betaIG(mu,var):
    aa=np.multiply(mu,np.true_divide(np.power(mu, 2),var)+1);
    return aa

def alphaGm(mu,var):
    aa=np.true_divide(np.power(mu, 2),var);
    return aa

def betaGm(mu,var):
    aa=np.true_divide(var,mu);
    return aa

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
