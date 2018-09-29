function [mix1]=SIN_init_VB_MM(data,opts)
%priors are more flat than in version 1

%SIN_init_VB_MM does
%              - fit a mixture model using ML (EM + MM algorithms (mmfit.m))
%              - initialize VB parameters of mixture model using EM fit as
%              initial posteriors
%INPUT: DATA is a data vector;
%       opts is a structure with fields
%           - MM: type of mixture model to fit 'GGM 'or 'GIM': [default 'GIM']
%           - K: number of components in mixture model: [default 3]
%           - comp: binary vector 1x3 marking which components are active; [default [1 1 1]] 
%                       - note: components are [0 + -] so for ex. comp=[1 1 0] fits a 2 component mixmod
%                           with a close to zero and a postive component.
%            -initialization: 'kmeans' or 'givens': [default kmeans]
%                       - selecting 'givens' requires to identify inits (see next)
%           - inits: matrix 3x2 containg 3 means and three variances, one per
%                 component;
%           - MLMMtol: tolerance for convergence of EM algorithm. [default 10^-5]
%            - MLMMits: maximum number of iterations performed by EM algorithm. [default 1]
%OUTPUT: MIX1 is a structure containing 
%           - mix1 is input for SIN_VB_MixMod.
%           - all the initialization parameters for the VB mixture model (priors and posteriors) 
%           - the ML solution (mix1.ML)
%EXAMPLE 1: 
%           Initialize a GGM for VB learning
%           Just fit a ML model (ignore the VB part of the output)
%           opts=[]; opts.MLMMits=100; opts.MLMMtol=10^-5;opts.MM='GGM';
%           mixML=SIN_init_VB_MM(data,opts); 


if ~isfield(opts,'MM'), MM ='GIM' ; 
else MM = opts.MM; end
if ~isfield(opts,'K'), K = 3; 
else K = opts.K; end
if ~isfield(opts,'comp'), comp = [1 1 1]; 
else comp = opts.comp; end
%no_comp=setdiff(1:3,find(comp));
if ~isfield(opts,'initialization'), initialization ='kmeans' ; end
if ~strcmp(opts.initialization,'kmeans'); initialization = opts.MLinit; end
if ~isfield(opts,'MLMMits'), MLMMits = 1; else
MLMMits= opts.MLMMits;    end
if ~isfield(opts,'MLMMtol'), MLMMtol = 10^-5; else
MLMMtol= opts.MLMMtol;    end



%Define method of moments functions for Gm and IGm
alphaIG=@(mu,var)(mu.^2./var)+2;
betaIG=@(mu,var)(mu.*((mu.^2./var)+1));
alphaGm=@(mu,var)(mu.^2./var);
betaGm=@(mu,var)(var./mu);


%SET PRIORS
%set mixing priors.
mix1.prior.lambda_0=5;

	%COMPONENT 1
	%set GAUSSIAN  prior on mean 
	mix1.prior.m_0=0;%init_param(1,1);%0;
	mix1.prior.tau_0=100;%000;%1/var(init_param(:,1));%10

	%set GAMMA prior on precission (shape and scale)
    mix1.prior.c0=.001;%shape
 	mix1.prior.b0=100;%scale

% 	%COMPONENTS 2 AND 3: gamma or inverse gamma
mix1.prior.d_0=0.001;
mix1.prior.e_0=0.001;

mix1.prior.loga_0=0;
mix1.prior.b_0=0.001;
mix1.prior.c_0=0.001;

%     mmm=10;%(the mean of component)
%     vvv=10;%(the variance of the component)
% 	if MM =='GGM' 
% %         %set GAMMA prior on rates (shape and rate)
%             Erate= 1/betaGm(mmm,vvv);
%             %Erate=betaIG(mmm,vvv);
%             mix1.prior.d_0=Erate;%d_0 = gamma shape
%             mix1.prior.e_0=1;%e_0 = gamma rate
%             
% %             mix1.prior.d_0=(Erate^2)/100;
% %             mix1.prior.e_0=Erate/100;%e_0 = gamma rate
%             
%             Erate=mix1.prior.d_0/mix1.prior.e_0;
%             %var_Erate=mix1.prior.d_0/(mix1.prior.e_0^2) ==> var_Erate = Erate
% 
%         %set shapes conditional prior (fancy)
%             Eshape=alphaGm(mmm,vvv);
%             dum_v=Eshape;%allow variance on shape to be of size of mean shape
%             dum_p=1/dum_v;
%             b_0=dum_p/psi(1,Eshape);%from laplace approx b=prec/psi'(map(s))
%             c_0=b_0;
%             loga_0=((b_0* psi(Eshape))-(c_0*log(Erate))); 
%             mix1.prior.loga_0=loga_0;
%             mix1.prior.b_0=b_0;
%             mix1.prior.c_0=c_0;
% 
% 
% 	elseif MM=='GIM'
% %         %set GAMMA prior on scale (shape d and rate e)
%         Escale=betaIG(mmm,vvv);
%             mix1.prior.d_0=Escale;%shape
%             mix1.prior.e_0=1;%rate
%             
% %             mix1.prior.d_0=(Escale^2)/100;
% %             mix1.prior.e_0=Escale/100;%e_0 = gamma rate
%             
% %             mix1.prior.d_0=100;
% %             mix1.prior.e_0=1;%e_0 = gamma rate
%             Escale=mix1.prior.d_0/mix1.prior.e_0;
%             %var_Escale=mix1.prior.d_0/(mix1.prior.e_0^2) ==> var_Escale =Escale
% 
%          %set component 2 and 3 shape conditional prior (fancy)
%             Eshape=alphaIG(mmm,vvv);
%             dum_v=Eshape;%allow variance on shape to be of size of mean shape
%             dum_p=1/dum_v;
%             b_0=dum_p/psi(1,Eshape);%from laplace approx b=prec/psi'(map(s))
%             c_0=b_0;
%             loga_0=(-(b_0* psi(Eshape))+(c_0*log(Escale))); 
%             mix1.prior.loga_0=loga_0;
%             mix1.prior.b_0=b_0;
%             mix1.prior.c_0=c_0;        
% 
% 	end


%SET POSTERIORS initializations using ML mixture models
 opts.K=K;
 opts.tol=MLMMtol;
 opts.MaxIts=MLMMits;
 opts.true_comp=comp;
 if MM=='GGM'
    opts.mmtype=2;
 elseif MM=='GIM'
   opts.mmtype =3;
 else
    opts.mmtype =1;%dummy, never used gmm 
end
if opts.initialization =='kmeans'
    km=kmeans1(K,data);
    opts.MLPriorPi=km.pi;
    %opts.MLPriorPi=[1/3 1/3 1/3];
    opts.MLinit=[0 km.v(2); max(km.m) km.v(3); -min(km.m) km.v(1)];
	%[init_param outpi Exp_likel Real_likel cost resp]=mmfit(data,opts);
    [MLMM]=mmfit(data,opts);
else
    opts.MLPriorPi=[1/3 1/3 1/3];
    opts.MLinit=initialization;
	%[init_param outpi Exp_likel Real_likel cost resp]=mmfit(data,opts);
    [MLMM]=mmfit(data,opts);
end
%mix1.ML.init=init_param;mix1.ML.pi=outpi;mix1.ML.Exp_likel=Exp_likel;
init_param=MLMM.init;
outpi=MLMM.pi;
resp=MLMM.resp;
mix1.ML=MLMM;
mix1.MLunwprobs=MLMM.unwprobs;
%INIT POSTERIORS BASED IN ML MIX MODEL
[dum b]=max(resp);
mix1.post.q=b;
mix1.post.gammas=resp;
mix1.post.lambda=sum(resp,2)';
mix1.post.pi=outpi;
	%COMPONENT 1: Gaussian component
		%hyperparam. on mean
		    mix1.post.m0=init_param(1,1);
		    mix1.post.tau0=1/mean(init_param(find(comp),1));

		%hyperparam. on precission
		    init_prec=1/init_param(1,2);
		    init_var_prec=var(1./init_param(find(comp),2));
            if numel(find(comp))==1
                init_var_prec=1;
            end
		    mix1.post.c0=alphaGm(init_prec,init_var_prec );%shape
		    mix1.post.b0=betaGm(init_prec,init_var_prec );%scale

	%COMPONENTS 2 AND 3: gamma or inverse gamma
		if MM=='GGM'
			%hyperparam. on rates
			    init_rates=[  1/  betaGm(abs(init_param(2,1)), init_param(2,2))    1/  betaGm(abs(init_param(3,1)), init_param(3,2))  ]  ;
			    dum_var_r= (init_rates)* 0.1;%    var(init_rates);
				mix1.post.d_0=alphaGm(init_rates,dum_var_r);%shape
				mix1.post.e_0=1./betaGm(init_rates,dum_var_r);%rate
			    Erates=mix1.post.d_0./mix1.post.e_0; % == init_rates


			%hyperparam. on shapes
			    init_shapes=[  alphaGm(abs(init_param(2,1)), init_param(2,2))    alphaGm(abs(init_param(3,1)), init_param(3,2))  ]  ;
			    %b_0=[1 1];c_0=b_0;  
                b_0=sum(resp(2:3,:),2)';c_0=b_0;  
			    loga_0=((b_0.* psi(init_shapes))-(c_0.*log(Erates))); 
			    %MAP_shapes=invpsi((loga_0+ (c_0 .* log(Erates))) ./ b_0) % == init_shapes
				mix1.post.loga_0=loga_0;
				mix1.post.b_0=b_0;
				mix1.post.c_0=c_0;
		elseif MM=='GIM'
			%hyperparam. on scales (inverse gamma) --> scale is r in the text,
			%r ~ inv gamma distr
			    init_scales=[  betaIG(abs(init_param(2,1)), init_param(2,2))    betaIG(abs(init_param(3,1)), init_param(3,2))  ]  ;
			    dum_var_sc=(init_scales)* 0.1;%var(init_scales);
                mix1.post.d_0=alphaGm(init_scales,dum_var_sc);%gamma shape
				mix1.post.e_0=1./[ betaGm(init_scales,dum_var_sc)];%gamma rate
			    Escales=mix1.post.d_0./ (mix1.post.e_0) ; % == init_scales
                
			%hyperparam. on shapes
			    init_shapes=[  alphaIG(abs(init_param(2,1)), init_param(2,2))    alphaIG(abs(init_param(3,1)), init_param(3,2))  ]  ;
			    %b_0=[1 1];c_0=b_0;  
			    sumgam=sum(resp,2);
			    b_0=sumgam(2:3)'; c_0=b_0;  
			    loga_0=(-(b_0.* psi(init_shapes))+(c_0.*log(Escales))); 
			    %MAP_shapes=invpsi((-loga_0+ (c_0 .* log(Escales))) ./ b_0) % == init_shapes
				mix1.post.loga_0=loga_0;
				mix1.post.b_0=b_0;
				mix1.post.c_0=c_0;
		else
			fprintf('Unrecognized mixture model type \n')
			fprintf('It must be one of GGM or GIM \n')
			%break
		end

%Save posterior expectations for initialization of VB mixModel
mix1.gammas=resp;
mix1.lambda=sum(resp,2);
mix1.pi=outpi;
mix1.mu1=init_param(1,1);
mix1.tau1=1/init_param(1,2);
mix1.q=mix1.post.q;
if MM=='GGM'
	mix1.shapes=alphaGm(init_param(2:3,1), init_param(2:3,2))';
	mix1.rates= 1./  betaGm(init_param(2:3,1), init_param(2:3,2))' ;
elseif MM=='GIM'
    mix1.shapes=[alphaIG(init_param(2:3,1), init_param(2:3,2))]'; 
	mix1.scales=  [betaIG(init_param(2:3,1), init_param(2:3,2))]';%   betaIG(init_param(3,1), init_param(3,2))  ];
end



mix1.opts=opts;
end

