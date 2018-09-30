function [src]=SIN_VB_MixMod(x,mix1)
%Vatiational Bayes Mixture model: Gauss/Gamma/Inverse Gamma
%INPUT: x is a data vector;
%       mix1 is output of [mix1]=SIN_init_VB_MM(data,opts)
%OUTPUT: src is a structure with the parameters of the fitted model

%---SET DEFAULT OPTIONS
if ~isfield(mix1.opts,'K'), K = 3; 
else K = mix1.opts.K; end
if ~isfield(mix1.opts,'comp'), comp = [1 1 1]; %componets [0 + -] active or not active
else comp = mix1.opts.comp; end
if ~isfield(mix1.opts,'MM'), MM='GIM'; 
else MM = mix1.opts.MM; end
if ~isfield(mix1.opts,'tol1'), tol1=10^-5;%  tol for conv.: relative change of NFE 
else tol1 = mix1.opts.tol1; end
if ~isfield(mix1.opts,'tol2'), tol2=10^-5;%  tol for conv.: relative change of NFE 
else tol2 = mix1.opts.tol2; end
if ~isfield(mix1.opts,'MaxNumIt'), numit=200;%  tol for conv.: relative change of NFE 
else numit = mix1.opts.MaxNumIt; end
if ~isfield(mix1.opts,'MaxNumIt2'), numit2=200;%  tol for conv. of subloop
else numit2 = mix1.opts.MaxNumIt2; end
%---SET DEFAULT OPTIONS


src=mix1;



%---SET PRIORS
% mixture prior
lambda_0=src.prior.lambda_0;

% component 1 precision priors
b0=src.prior.b0;%scale
c0=src.prior.c0;%shape

% component 1 mean priors
m_0=src.prior.m_0;
tau_0=src.prior.tau_0;

% component 2 and 3  rate priors
d_0=src.prior.d_0; %shape
e_0=src.prior.e_0; %rate

% component 2 and 3 shape priors
loga_0=src.prior.loga_0;
b_0=src.prior.b_0;
c_0=src.prior.c_0;
%---SET PRIORS





pos=find(x>0);neg=find(x<0);
xpos=x;xpos(neg)=0;
xneg=x;xneg(pos)=0;
xneg=-xneg;
xx=[xpos ;xneg]; %I made all positive!!



% --Initialise usefull vector values
ftot=0;
%Fgauss = 0;
FEs=zeros(1,numit);
FEs(1:2)=[-10^40 -10^39];
likelihood=zeros(1,numit);
kls=zeros(1,numit);
ents=zeros(1,numit);
flag=0;
it=0;
%--Initialise usefull vector values


% --------   ITERATE POSTERIORS
while flag==0 
it=it+1;

%==================================E_Step================================    
    
    % compute ´responsibilitites´
    [gammas] = my_gammas(src,x,MM,comp);	
%==================================E_Step================================   

%==================================M_Step================================
        gamma_sum = sum(gammas');
        [~, dum]=max(gammas);
        src.q=dum;
    %--------------------Update lambda etc.--------------------
        lambda = lambda_0+gamma_sum;

    
            % store for E-step
            src.post.lambda=lambda;

            % contribution to energy  (copied from choud)  
            lambda_p=lambda_0*ones(1,3);
            dir1 = sum(gammaln(lambda+eps) - gammaln(lambda_p+eps));
            dir2 = gammaln(sum(lambda+eps)) - gammaln(sum(lambda_p+eps));
            %Fdir = dir1-dir2;
            ent_gam=-sum(sum(gammas.*log(gammas+eps)));
                %alb-->
                dir3=sum( (lambda-lambda_0) .* (psi(lambda)-psi(sum(lambda))));
                KLPI=dir2-dir1+dir3;
            
    %--------------------Update lambda etc.--------------------
 

%--------------------Update component 1 ---------------------   
     pdf_fact=1/2;  
             %--------------------Update precision tau1---------------------
                  mu1=src.mu1;
                  tau1=src.tau1;
                  tau = tau_0+(tau1.*gamma_sum(1));
                  mean_xsq = sum(gammas(1,:).*(x.^2));
                  mean_x = sum(gammas(1,:).*x);
                  %mu_sq = (gamma_sum(1).*(mu1.^2+(1./tau1)))';
                  mu_sq = (gamma_sum(1).*(mu1.^2+(1./tau)))';%!!!!!!!
                  data_bit = mean_xsq-2*mu1'.*mean_x+mu_sq;
                  b = 1./( (1/b0)  + (pdf_fact*data_bit'));
                  c = c0+(pdf_fact*gamma_sum(1));
                  mean_tau1 =b*c;

                    % store for E-step
                        src.post.b0=b;
                        src.post.c0=c;
                    % contribution to energy using KL of  Gamma, check b=scale and c=shape
                        bp=b0;cp=c0;
                        bq=b;cq=c;
                        KLTAU1= (cq*((bq/bp)-1)) + ((cq-cp)*(psi(cq) + log(bq))) - gammaln(cq) - ( cq*log( bq))   + gammaln(cp) + (cp*log( bp));
                      
              %--------------------Update precision---------------------

 		%-----------------------Update mean-----------------------
                    tau = tau_0+(mean_tau1.*gamma_sum(1));
                    %mm = 1./tau.*(m_0+mean_tau1.*mean_x'); typo in choud code??
                    mm = 1./tau.*( (tau_0*m_0) + (mean_tau1.*mean_x'));
                    mean_mu1=mm;
                    mean_mu12=mm.^2+(1/tau);
                    
                        % store for E-step
                            src.post.m0=mm;
                            src.post.tau0=tau;
                        % contribution to energy using KL of Gauss
                           bp=tau_0;mp=m_0;
                           bq=mean_tau1;mq=mean_mu1;
                           KLMU1=1/2 *(  ((bp/bq)-1)-log(bp/bq)+ (bp*((mq-mp)^2)  ));
                  %-----------------------Update means-----------------------

  %--------------------Update components 2 and 3 ---------------------   
                if it==1
                    if strcmp(MM,'GGM')
                            MAP_shape=src.shapes;

                    elseif strcmp(MM,'GIM')
                            MAP_shape=src.shapes;
                    end
                end
                subflag=0; %for convergence of Gamma/Inverse Gamma parameters iteration
                its2=0;
                while subflag==0  %for its2=1:subloop
                    its2=its2+1;
                        %--------------------Update rates if GGM or scales if GIM---------------------
                        if strcmp(MM,'GGM')
                                    mean_x =sum(gammas(2:3,:).*xx,2);
                                    e=e_0 + mean_x';
                                    d=d_0+(MAP_shape.*gamma_sum(2:3) );
                                    mean_rates=d./e;
                                    mean_logrates=psi(d)-log(e);
                        elseif strcmp(MM,'GIM')%MM =='GIM'%conj prior on rate of inv gamma is gamma
                                    tmp=gammas(2:3,:)./xx;
                                    mean_x(1)=sum(tmp(1,pos));
                                    mean_x(2)=sum(tmp(2,neg));
                                    mean_x=mean_x';
                                    e=e_0 + mean_x';
                                    d=d_0+(MAP_shape.*gamma_sum(2:3) );
                                    clear mean_x                                    
                                    mean_scales=d./e;
                                    mean_logscales=psi(d)-log(e);            
                        end     
                        %--------------------Update rates if GGM or scales if GIM---------------------


                        %--------------------Update shapes---------------------
                        B=  b_0+ gamma_sum(2:3); 
                        C=  c_0+ gamma_sum(2:3); 
                        xresp=xx.^gammas(2:3,:);%.*xx;  
                        idx= xresp==0;
                         xresp(idx)=1;
                        logA=loga_0 +sum(log(xresp'));     
                        if strcmp(MM,'GGM')
                            MAP_shape=invpsi((logA+ (C .* mean_logrates)) ./ B);
                        elseif strcmp(MM,'GIM')
                            MAP_shape=invpsi((-logA+ (C .* mean_logscales)) ./ B) ;
                        end
                        %--------------------Update shapes--------------------- 
                        
                        %-----check convergence of subloop
                        if its2>1
                            if strcmp(MM,'GGM')
                                new=[mean_rates MAP_shape];%dummm variable for testing convergence   
                            elseif strcmp(MM,'GIM')
                                new=[mean_scales MAP_shape];%dummm variable for testing convergence
                            end 
                            mean_rel_change=mean(abs(old-new)./old);
                            if (its2>numit2) || mean_rel_change<tol2
                                subflag=1;
                            end
                        end
                        if strcmp(MM,'GGM')
                            old=[mean_rates MAP_shape];%dummm variable for testing convergence       
                        elseif strcmp(MM,'GIM')
                            old=[mean_scales MAP_shape];%dummm variable for testing convergence
                        end 
                        %-----check convergence of subloop
                        
                        clear tmp

                end
                 % store for E-step
                     src.post.d_0=d;
                     src.post.e_0=e;
                     
                     src.post.loga_0=logA;
                     src.post.b_0=B;
                     src.post.c_0=C;
                 %contribution to energy 
                 f2=@(b,alpha) (b .* psi(1,alpha));% from Laplace approx
                 tmpidx=find(comp(2:3));
                if strcmp(MM,'GGM')
                        %using KL of  Gamma, check b=scale and c=shape?
                       bp=1./e_0;cp=d_0;
                       bq=1./e;cq=d;                                   
                       KLrates= (cq.*((bq./bp)-1)) + ((cq-cp).*(psi(cq) + log(bq))) - gammaln(cq) - ( cq.*log( bq))   + gammaln(cp) + (cp.*log( bp));
                       KLrates=KLrates(tmpidx);
                       KLrate=sum(KLrates);
                       
                       
                        % using KL of  Gauss and Laplace approx
                        logprExpected_rate=psi(d_0)-log(e_0);%%!!!!!!!get out of loop
                        prmean=invpsi((loga_0+ (c_0 * logprExpected_rate)) / b_0); %!!!!!!!get out of loop
                        prprecc=f2(b_0,prmean);%!!!!!!!get out of loop
                        bp=prprecc;mp=prmean;%!!!!!!!get out of loop
                        bq=f2(B,MAP_shape);mq=MAP_shape;
                        KL_fancys=1/2 *(  ((bp./bq)-1)-log(bp./bq)+ (bp*((mq-mp).^2)  )); 
                         KL_fancys=KL_fancys(tmpidx);
                        KL_fancy=sum(KL_fancys);
                       
                elseif strcmp(MM,'GIM')
                        %using KL of  Gamma, check b=scale and c=shape?
                        bp=1./e_0;cp=d_0;
                        bq=1./e;cq=d;    
                        KLscales= (cq.*((bq./bp)-1)) + ((cq-cp).*(psi(cq) + log(bq))) - gammaln(cq) - ( cq.*log( bq))   + gammaln(cp) + (cp.*log( bp));
                        KLscales=KLscales(tmpidx);
                        KLscale=sum(KLscales);  
                        
                        
                        % using KL of  Gauss and Laplace approx
                        logprExpected_scale =psi(d_0)-log(e_0);%%!!!!!!!get out of loop
                        prmean=invpsi((-loga_0+ (c_0 * logprExpected_scale)) / b_0); %%!!!!!!!get out of loop
                        prprecc=f2(b_0,prmean);%%!!!!!!!get out of loop
                        bp=prprecc;mp=prmean;%%!!!!!!!get out of loop
                        bq=f2(B,MAP_shape);mq=MAP_shape;
                        KL_fancys=1/2 *(  ((bp./bq)-1)-log(bp./bq)+ (bp.*((mq-mp).^2)  )); 
                        KL_fancys=KL_fancys(tmpidx);
                        KL_fancy=sum(KL_fancys); 
                end
                                       
                       
 

                
%==================================M_Step================================        

%==================================Energy================================
%compute tilde pi
tildepi=exp(psi(lambda)-psi(sum(lambda)));
tildetau1=b*exp(psi(c));
%ggm: tilder=(1./e).*exp(psi(d));
%gim: tilder=(1./e).*exp(psi(d));
%FE expected_likelihood contribution (L)

    %gauss component
    const_bit=gammas(1,:)*(log(tildepi(1))+(log(tildetau1)/2));
    data_bit= gammas(1,:).*((mean_tau1/2)*(x.^2 + mean_mu12 -(2*mean_mu1.*x)));
    L(1)=sum(const_bit-data_bit)-((sum(gammas(1,:))/2)*log(2*pi));%((numel(x)/2)*log(2*pi));
    %non-gauss components 
    xxx=xx;
    xxx(xxx==0)=1;
    ElogGams=gammaln(MAP_shape)+ (1./B) +   (  (psi(2,MAP_shape).*MAP_shape)./ (psi(1,MAP_shape).*B )   );
    %ElogGams=gammaln(MAP_shape)+ (1./B)+   ((psi(2,MAP_shape).*MAP_shape)./ (psi(1,MAP_shape).*B )   ); SAME FOR GGM AND GIM ??
        if strcmp(MM,'GGM')
            data_bit=gammas(2:3,:).* ( (repmat((MAP_shape'-1),1,numel(x)).*log(xxx)) - (repmat((mean_rates'),1,numel(x)).*xx));
            const_bit=(log(tildepi(2:3))+ (MAP_shape.*mean_logrates)-ElogGams)';
            const_bit= gammas(2:3,:).*  repmat(const_bit  , 1,numel(x)); 
            %L(2:3)=sum(const_bit+data_bit,2);
         elseif strcmp(MM,'GIM')
            data_bit=gammas(2:3,:).* ((repmat((MAP_shape'+1),1,numel(x)).*-log(xxx)) - (repmat(mean_scales',1,numel(x))./xxx));
            const_bit=(log(tildepi(2:3))+ (MAP_shape.*mean_logscales)-ElogGams)';
            const_bit= gammas(2:3,:).*  repmat(const_bit  , 1,numel(x));
        end
        tmpidx=find(comp(2:3));
        data_bit=data_bit(tmpidx,:);
        const_bit=const_bit(tmpidx,:);
        tmp=const_bit+data_bit;
        L(2)=sum(sum(tmp,2));
        LIK=sum(L);
            
            
%FE other contributions
if strcmp(MM,'GGM')
    KLs=KLPI+KLMU1+KLTAU1+KLrate+KL_fancy;
 elseif strcmp(MM,'GIM')
    KLs=KLPI+KLMU1+KLTAU1+KLscale+KL_fancy; 
end


f_hidd = ent_gam;%-numel(x)/2*log(2*pi);
ftot=LIK+ent_gam-KLs;
FEs(it)=ftot;
likelihood(it)=LIK;
kls(it)=-KLs;
ents(it)=f_hidd;

%check convergence
if it>1
    progress(it)=abs(FEs(it)-FEs(it-1))/abs(FEs(it));
    if it>numit ||( progress(it) <tol1)
        flag=1;
    end
end
%it
end
FEs=FEs(1:it);
likelihood= likelihood(1:it);
kls= kls(1:it);
ents= ents(1:it);

%==================================Energy================================	



src.lambda=lambda;
%src.pi = lambda./sum(lambda);
src.gammas=gammas;
src.pi=(sum(gammas,2)/sum(sum(gammas)))';
src.mu1=mean_mu1;
src.tau1=mean_tau1;
src.shapes=MAP_shape;

src.ftot=ftot;
src.FEs=FEs;
src.pi;
src.LIK=likelihood;
src.kls=kls;
src.ents=ents;

if strcmp(MM,'GGM')
	src.rates=mean_rates;
elseif strcmp(MM,'GIM')
	src.scales=mean_scales;
end

src.it=it;%number of iterations

end 


function [resp] = my_gammas(src,data,MM,true_comp)
pos= data>0;
neg= data<0;

%expectation on log pi
    dum=src.post.lambda;
    ElogPi=psi(dum)-psi(sum(dum));
%expectations for component 1
    %expectations on mu1 and mu1^2
        dum1=src.post.m0;
        dum2=src.post.tau0;
        Emu1=dum1;
        Emu12=dum1^2+ (1/dum2);
    %expectations on tau1 and log tau1
        dum1=src.post.c0;%shape
        dum2=src.post.b0;%scale
        Etau1=dum1*dum2;
        Elogtau1=psi(dum1)+log(dum2);         
%expectations for component 2 and 3
    %expectations on r and log r (r=rate for GGM and scale for GIM)
        dum1=src.post.d_0;%shape
        dum2=src.post.e_0;%rate 
        if strcmp(MM,'GGM')
            Er=dum1./dum2;
            Elogr=psi(dum1)-log(dum2);
        elseif strcmp(MM,'GIM')
            Er=dum1./dum2;
            Elogr=psi(dum1)-log(dum2);
        end
    %expectations on s and ? log(gamma(s))?? (s=shape)
        dum1=src.post.loga_0;
        dum2=src.post.b_0;
        dum3=src.post.c_0;
        if strcmp(MM,'GGM')
            Es=invpsi((dum1+ (dum3 .* Elogr)) ./ dum2);
        elseif strcmp(MM,'GIM')
            Es=invpsi((-dum1+ (dum3 .* Elogr)) ./ dum2);
        end
        %approximation to E[log(gamma(s))]
        ElogGams=gammaln(Es) + (1./dum2) +   (  (psi(2,Es).*Es)./ (psi(1,Es).*dum2 )   );   
             
        
%compute 'responsibilities'
if strcmp(MM,'GGM')
    resp(1,:)= exp(    ElogPi(1)+(Elogtau1/2)-(log(2*pi)/2) -( (data.^2 + Emu12-(2.*Emu1.*data) )* (Etau1/2) )     );
    %resp(2,:)=exp(    ElogPi(2)+ (repmat(Es(1)-1,1,numel(data)).*log(data)) +(Es(1).*Elogr(1)) - (ElogGams(1)) - (repmat(Er(1),1,numel(data)).*data)   );
    %resp(3,:)=exp(    ElogPi(3)+ (repmat(Es(2)-1,1,numel(data)).*log(-data)) +(Es(2).*Elogr(2)) - ( ElogGams(2)) - (repmat(Er(2),1,numel(data)).*-data)   );
    dum=exp(  repmat(ElogPi(2:3)',1,numel(data))+ (repmat(Es'-1,1,numel(data)).* [log(data); log(-data)]) +repmat((Es.*Elogr)',1,numel(data)) - ( repmat(ElogGams',1,numel(data))) - (repmat(Er',1,numel(data)).* [data; -data]));
    resp(2:3,:)=dum;
    clear dum

elseif strcmp(MM,'GIM')
    resp(1,:)= exp(    ElogPi(1)+(Elogtau1/2)-(log(2*pi)/2) -( (data.^2 + Emu12-(2.*Emu1.*data) )* (Etau1/2) )     );   
    %resp(2,:)=exp(    ElogPi(2)- (repmat(Es(1)+1,1,numel(data)).*log(data)) +(Es(1).*Elogr(1)) - (ElogGams(1)) - (repmat(Er(1),1,numel(data))./data)   );
    %resp(3,:)=exp(    ElogPi(3)- (repmat(Es(2)+1,1,numel(data)).*log(-data)) +(Es(2).*Elogr(2)) - ( ElogGams(2)) - (repmat(Er(2),1,numel(data))./-data)   );
    dum=exp(  repmat(ElogPi(2:3)',1,numel(data))- (repmat(Es'+1,1,numel(data)).* [log(data); log(-data)]) +repmat((Es.*Elogr)',1,numel(data)) - ( repmat(ElogGams',1,numel(data))) - (repmat(Er',1,numel(data))./ [data; -data]));
    resp(2:3,:)=dum;
    clear dum
end
no_comp=setdiff(1:3,find(true_comp));
resp(no_comp,:)=0;    
resp(2,neg)=0;
resp(3,pos)=0;
     
resp=resp./repmat(sum(resp),3,1);
end














function Y=invpsi(X)
    % Y = INVPSI(X)
    %
    % Inverse digamma (psi) function.  The digamma function is the
    % derivative of the log gamma function.  This calculates the value
    % Y > 0 for a value X such that digamma(Y) = X.
    %
    % This algorithm is from Paul Fackler:
    % http://www4.ncsu.edu/~pfackler/
    %
      L = 1;
      Y = exp(X);
      while L > 10e-8
        Y = Y + L*sign(X-psi(Y));
        L = L / 2;
      end
end


