clear
addpath(genpath('~/Dropbox/POSTDOC/MY_POSTDOC_TOOLBOX/code'))
addpath(genpath('C:\Users\alblle\Dropbox\POSTDOC\MY_POSTDOC_TOOLBOX\code'))
%fix seed
%sprev = rng(0,'v5uniform'#randn

load data_vector.mat
data=data_vector;
%==========MIXTURE MODEL FIT OPTIONS===============  
%Number of components to fit 
options.K=3; 
if options.K==3
    options.comp=[1 1 1];% [gauss posit negat] components.
elseif options.K==2
    options.comp=[1 1 0]; %or [1 0 1];
elseif options.K==1    
    options.comp=[1 0 0]; 
end            
            
%convergence options
options.tol1=10^-5;%  main loop tolerance for conv.: relative change of NFE 
options.tol2= 10^-5;%options.tol1;%10^-5;%   secundary loop tolerance for conv.: relative change of NFE 
options.MaxNumIt=200;
options.MaxNumIt2=1;
%options.MLMMits=1;
options.MLMMtol=options.tol1;%10^-5;
            
%==========MIXTURE MODEL FIT OPTIONS===============  


     



%initialization? 
if 0
    options.initialization ='kmeans';
else
    options.initialization ='givens';
    ms=[0 3 -3]';
    vs=[1 2 2]';
    options.MLinit=[ms vs]; 
    if options.K==3
        %[0 1; 2.7 1; -2.7 1];  
    else
        options.inits=options.MLinit.*repmat(options.comp',1,2);
    end
   
            end

        
%==========FIT MIXTURE MODEL USING EM WITH METHOD OF MOMENTS
        options2=options;
        options2.MLMMits=10000;
        

%==========FIT MIXTURE MODEL USING EM WITH METHOD OF MOMENTS


        
% %==========FIT MIXTURE MODEL USING VB
%         %========== FIT  VB-GGM ===============  
%                 options.MM='GGM';
%                 options.MLMMits=1;
%         %         options.comp=[1 0 1]; %or [1 0 1];
%                 tic
%                 mix1=SIN_init_VB_MM(data,options); 
%                 src{1}=SIN_VB_MixMod(data,mix1)  ;
%                 time(1)=toc;
%                 cost(1)=src{1}.it;
%                 fprintf('VB-GGM took %d seconds (%d iterations) \n',time(1),cost(1))
%         %========== FIT  VB-GGM ===============  


        %========== FIT  VB-GIM===============       
                options.MM='GIM';
                options.MLMMits=1;
                tic
                mix2=SIN_init_VB_MM(data,options);        
                src{2}=SIN_VB_MixMod(data,mix2)  ;
                time(2)=toc;
                cost(2)=src{2}.it;
                fprintf('VB-GIM took %d seconds (%d iterations) \n',time(2),cost(2) )
        %========== FIT  VB-GIM===============  
%==========FIT MIXTURE MODEL USING VB

%         %========== FIT  ML-GGM ===============  
%                tic
%                 options2.MM='GGM';
%                 mix1=SIN_init_VB_MM(data,options2); 
%                 time(3)=toc;
%                 src{3}=mix1 ;
%                 cost(3)=numel(src{3}.ML.Exp_likel);
%                 fprintf('ML-GGM took %d seconds (%d iterations) \n',time(3), cost(3))
%         %========== FIT  ML-GGM ===============  
% 
%         %========== FIT  ML-GIM ===============  
%                tic 
%                 options2.MM='GIM';
%                 mix1=SIN_init_VB_MM(data,options2); 
%                 time(4)=toc;
%                 src{4}=mix1 ;
%                 cost(4)=numel(src{4}.ML.Exp_likel);
%                 fprintf('ML-GIM took %d seconds (%d iterations) \n',time(4), cost(4));
% 
%         %========== FIT  ML-GIM ===============  