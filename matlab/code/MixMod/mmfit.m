function [MLMM]=mmfit(x,opts)%K,init,mmtype,tol,MaxIts,PriorPi,true_comp)
%mmfit does
%              - fit a mixture model using ML (EM + MM algorithms (mmfit.m))
%INPUT: DATA is a data vector;
%       OPTIONS is a structure with fields
%           - (MUST) opts.MLinit  must be a matrix 3x2 containg 3 means and three variances, one per
%                 component;
%           - MM: type of mixture model to fit 'GGM 'or 'GIM': [default 'GIM']
%           - K: number of components in mixture model: [default 3]
%           - comp: binary vector 1x3 marking which components are active; [default [1 1 1]] 
%                       - note: components are [0 + -] so for ex. true_comp=[1 1 0] fits a 2 component mixmod
%                           with a close to zero and a postive component.
%           - MLMMtol: tolerance for convergence of EM algorithm. [default 10^-5]
%           - MLMMits: maximum number of iterations performed by EM algorithm. [default 200]
%           - PriorPi: EM init mixing proportions [default [.33 .33 .33]] 


if ~isfield(opts,'MLinit'), fprintf('No initialization provided in mmfit.m \n') 
else inits=opts.MLinit;end
if ~isfield(opts,'mmtype'),mmtype='3' ; %3GIM, 2GGm
else mmtype = opts.mmtype; end
if ~isfield(opts,'K'), K = 3; 
else K = opts.K; end
if ~isfield(opts,'comp'),true_comp=[1 1 1] ; 
else true_comp = opts.comp; end
if ~isfield(opts,'MLMMtol'),tol=10^-5 ; 
else tol = opts.MLMMtol; end
if ~isfield(opts,'MLMMits'),MaxIts=200 ; 
else  MaxIts= opts.MLMMits; end
if ~isfield(opts,'MLPriorPi'),PriorPi= [1/3 1/3 1/3]; 
else  PriorPi= opts.MLPriorPi; end
if ~isfield(opts,'shift'), shift=0 ; 
else shift = opts.shift; end


no_comp=setdiff(1:3,find(true_comp));

%define each component type of the mm
if mmtype==1
    presModel= ones(1,3);
elseif mmtype==2
    presModel= [1 2*ones(1,2)];
elseif mmtype==3
    presModel= [1 3*ones(1,2)];
end

pos_shift=find(x<shift);
neg_shift=find(x>-shift);
%set estimated parameters to initialization
stp=inits;
init_PI=PriorPi;%(1/K)*ones(1,K); 
tmpPI=init_PI;
        %first iteration
        allparam(1,:,:)=stp;
        allPi(1,:)=tmpPI;
                for comp=find(true_comp);%1:K
                    tmpdata=x;
                    if comp==3
                        tmpdata=-x;
                    end
                    [prob]= pdf_Ga_Gm_IG(tmpdata,stp(comp,1),stp(comp,2),presModel(comp));
                   
                     prob=prob*tmpPI(comp);
                    probs(comp,:)=prob;
                end
                probs(probs<10^-14)=eps;
                probs(no_comp,:)=eps;
                if shift~=0
                    probs(2,pos_shift)=eps;
                    probs(3,neg_shift)=eps;
                end
                GenResp= probs./repmat(sum(probs),3,1);
                GenResp (find(isnan(GenResp)))=10^-14;
                GenN=sum(GenResp,2);
                tmpPI=(GenN./sum(GenN))';%Gresp'/sum(Gresp');
                dum=log(repmat(tmpPI',1,size(probs,2)))+log(probs);
                Exp_likel(1)=sum( sum( GenResp.*dum   ));
                dum2=repmat(tmpPI',1,size(probs,2)).*probs;
                Real_likel(1)=sum(log(sum(dum2)));
                flag=0;
                it=2;
                while flag==0
                    for comp=1:2
                        %use repmat
                        stp(comp,1)=(sum(GenResp(comp,:).*x))/GenN(comp);%means update
                        stp(comp,2)=sum(GenResp(comp,:).*((x-stp(comp,1)).^2))/GenN(comp);%variances update
                    end
                    comp=3;
                    stp(comp,1)=(sum(GenResp(comp,:).*-x))/GenN(comp);%means update
                    stp(comp,2)=sum(GenResp(comp,:).*((-x-stp(comp,1)).^2))/GenN(comp);%variances update
                    
                    
                    for comp=find(true_comp);%K
                        tmpdata=x;
                        if comp==3
                            tmpdata=-x;
                        end
                        [prob]= pdf_Ga_Gm_IG(tmpdata,stp(comp,1),stp(comp,2),presModel(comp));
                        unwprobs(comp,:)=prob;
                         prob=prob*tmpPI(comp);
                         prob(find(isnan(prob)))=10^-16;
                        probs(comp,:)=prob;
                    end
                    probs(no_comp,:)=eps;
                    unwprobs(no_comp,:)=eps;
                    if shift~=0
                        probs(2,pos_shift)=eps;
                        probs(3,neg_shift)=eps;
                    end
                    
                    GenResp = probs./repmat(sum(probs),3,1);
                    %if supervised ==1; GenResp=supresp;end
                    GenResp(find(isnan(GenResp)))=10^-14;
                    GenN=sum(GenResp,2);
                    tmpPI=GenN./sum(GenN);
                    dum=log(repmat(tmpPI,1,size(probs,2)))+log(probs);
                    dum(dum==-Inf)=10^-14;
                    dum(dum==Inf)=10^-14;
                    Exp_likel(it)=sum( sum( GenResp.*dum   ));
                    dum2=repmat(tmpPI,1,size(probs,2)).*probs;
                    Real_likel(it)=sum(log(sum(dum2)));
                    allparam(it,:,:)=stp;
                    allPi(it,:)=tmpPI;
                    if it>2
                        err=abs(Real_likel(it-1)-Real_likel(it))/ abs(Real_likel(it-1)); 
                        %err2=Real_likel(it)>Real_likel(it-1);
                        if err<tol | it>MaxIts %|err2
                            flag=1;   
                        end
                    end
                    it=it+1;
                end
                cost=it;
                outparam=squeeze(allparam(end,:,:));
                outpi=squeeze(allPi(end,:));
                
MLMM=[];
MLMM.init=outparam;%init contains the results!! counterintuitive but ML, better for vb notation
MLMM.pi=outpi;
MLMM.Exp_likel=Exp_likel;
MLMM.Real_likel=Real_likel;
MLMM.cost=cost;
MLMM.resp =GenResp ;
MLMM.probs=probs;
MLMM. unwprobs =   unwprobs;     
               
end

function [probs]=pdf_Ga_Gm_IG(x,M,V,dist) 
    alphaIG=@(mu,var)(mu^2/var)+2;
    betaIG=@(mu,var)(mu*((mu^2/var)+1));
    alphaGm=@(mu,var)(mu^2/var); %gamma shape
    betaGm=@(mu,var)(var/mu);%gamma scale
    
    gam = @(x,a,b) (1/(b^a*gamma(a))).*(x).^(a-1).*exp(-x./b);
    myp1=@(x,M,V) normpdf(x,M,V^(1/2));
    myp2=@(x,M,V)gampdf(x,alphaGm(M,V),betaGm(M,V));
    myp3=@(x,M,V)invgam(x,alphaIG(M,V),betaIG(M,V));
    
    myp4=@(x,M,V)exp(alphaGm(M,V)*log(betaGm(M,V)) - log(gamma(alphaGm(M,V))) -((alphaGm(M,V)+1)*log(x)) -(betaGm(M,V)/x));
    
    if dist==0
        Gapr=myp1(x,M,V);
        Gmpr=myp2(x,M,V);
        IGmpr=myp3(x,M,V);
        probs=[Gapr; Gmpr; IGmpr];
    elseif dist==1
        Gapr=myp1(x,M,V); probs=[Gapr];
         probs=[Gapr];
    elseif dist==2
         Gmpr=myp2(x,abs(M),V); 
         probs=[Gmpr];
    elseif dist==3
        %a=alphaGm(abs(M),V);
        %b=betaGm(M,V);
        %myp4(x,M,V);
        IGmpr=myp3(x,abs(M),V);
        probs=[IGmpr];
    end
    probs(find(isnan(probs)))=10^-14;
    probs(probs==-Inf)=10^-14;%0;%hack!!
    probs(probs==Inf)=10^-14;

end

function p= invgam(x,a,b) 
    tmp=b^a/gamma(a).*(1./x).^(a+1).*exp(-b./x);
    idx=find(x<10^-14);
    tmp(idx)=10^-14;
    p=tmp;
end



