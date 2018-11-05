function [output]=mmbetas_2comp(data)%,opts)%K,init,mmtype,tol,MaxIts,PriorPi,true_comp)
model=1;
if model==1
    method_of_moment=1;
    max_lik=0;
elseif model==2
    method_of_moment=0;
    max_lik=1;
end

tol=10^-10;
MaxIts=500;

%define method of moments and max likelihood necesary functions
%MM for beta
%m=mean(x);
%s=std(x)^2;
a_MM=@(m,s)m*((m*(1-m)/s) -1);
b_MM=@(m,s)(1-m)*((m*(1-m)/s)-1) ;
% ML for the beta
g1=@(a,b,x) psi(a) - psi(a+b) - mean(log(x));
g2=@(a,b,x) psi(b) - psi(a+b) - mean(log(1-x));
dg1a=@(a,b) psi(1,a) - psi(1,a+b);
dg1b=@(a,b) - psi(1,a+b);
dg2a=@(a,b) - psi(1,a+b);
dg2b=@(a,b) psi(1,b) - psi(1,a+b);
g=@(a,b,x)[g1(a,b,x); g2(a,b,x)];
G=@(a,b)[dg1a(a,b) dg1b(a,b); dg2a(a,b) dg2b(a,b)];



% m=mean(data);
% s=std(data)^2;
% theta=[a_MM(m,s); b_MM(m,s)];
% flag=0;
% while flag==0
%     theta_old=theta;
%     theta=theta-(inv(G(theta(1), theta(2))) * g(theta(1),theta(2),data))
% end



%init parameters
init_PI=[.5 .5 ];
tmpPI=[.5 .5];
params{1}=[1 5];
params{2}=[5 1];
estimated_parameters=[params{1};params{2}];

%first iteration
allparam(1,:,:)=[params{1};params{2}];
allPi(1,:)=tmpPI;
for comp=1:2
    [prob]= betapdf(data,estimated_parameters(comp,1),estimated_parameters(comp,2)); 
    prob=prob*tmpPI(comp);
    probs(comp,:)=prob;
end
probs(probs<10^-14)=eps;
GenResp= probs./repmat(sum(probs),2,1);
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
    %PERFORM UPDATES OF BETA PARAMETERS
    for comp=1:2
        %use repmat
        if method_of_moment==1
            stm(comp)=(sum(GenResp(comp,:).*data))/GenN(comp);%means update
            stv(comp)=sum(GenResp(comp,:).*((data-stm(comp)).^2))/GenN(comp);%variances update
            estimated_parameters(comp,1)=a_MM(stm(comp),stv(comp));
            estimated_parameters(comp,2)=b_MM(stm(comp),stv(comp));
        elseif max_lik==1  
            theta=[estimated_parameters(comp,1); estimated_parameters(comp,2)];
            ml_sub_flag=0;
            ml_sub_its=0;
            while ml_sub_flag==0
                ml_sub_its=ml_sub_its+1;
                theta_old=theta;
                %theta2=theta-(inv(G(theta(1), theta(2))) * g(theta(1),theta(2),data))
                %Resp=GenResp(comp,:);N=GenN(comp);
                em_g1=@(a,b,x,Resp,N) psi(a) - psi(a+b) - (sum(Resp.*log(data)))/N;
                em_g2=@(a,b,x,Resp,N) psi(b) - psi(a+b) - (sum(Resp.*log(1-data)))/N;
                em_g=@(a,b,x,Resp,N)[em_g1(a,b,x,Resp,N); em_g2(a,b,x,Resp,N)];
                theta=theta-(inv(G(theta(1), theta(2))) * em_g(theta(1),theta(2),data,GenResp(comp,:),GenN(comp)));
                if norm(theta-theta_old)<10^-6
                    ml_sub_flag=1;
                end
            end 
            estimated_parameters(comp,1)=theta(1);
            estimated_parameters(comp,2)=theta(2);
        end
    end

    for comp=1:2
        [prob]= betapdf(data,estimated_parameters(comp,1),estimated_parameters(comp,2)); 
        unwprobs(comp,:)=prob;
         prob=prob*tmpPI(comp);
         prob(find(isnan(prob)))=10^-16;
        probs(comp,:)=prob;
    end

    GenResp = probs./repmat(sum(probs),2,1);
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
    allparam(it,:,:)=estimated_parameters;
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
                
output=[];
output.beta_params=outparam;
output.pi=outpi;
output.Exp_likel=Exp_likel;
output.Real_likel=Real_likel;
output.cost=cost;
output.resp =GenResp ;
output.probs=probs;
output. unwprobs =   unwprobs;     
               
end

