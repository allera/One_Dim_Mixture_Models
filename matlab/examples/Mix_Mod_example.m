%Example of 1-D mixture model fit
%uses mix=SIN_init_VB_MM(data,options) and src=SIN_VB_MixMod(data,mix);
%mix=SIN_init_VB_MM(data,options) initialize the model using method of
%moments and can be used to learn an approximated ML solution
%src=SIN_VB_MixMod(data,mix)uses the output of the init function and
%performs VB inference.

clear

%add code to path
addpath(genpath('../code/MixMod/'))
addpath(genpath('../code/needs/'))

%==========1-dimensional, 3 components, mixture model fit OPTIONS ===============
%K = number of components to fit 
    options.K=3;
%MM defines the mixture model type, GGM or GIM,
    options.MM='GIM';
%MLMMits = number of ML iterations at init of VB  
    options.MLMMits=1;
%Initialization:'givens'or 'kmeans'
    options.initialization='givens'; % other choice is 'kmeans'
 
%options.comp=[1 0 1]; %or [1 0 1];
%==========1-dimensional, 3 components, mixture model fit OPTIONS ===============






%==========mixture model data generation ===============
    %datatype = MM type to generate data from: 1=GGM, 2=GGM, 3=GIM  
        datatype=3;   
    % N = number of samples
        N=10000;                
    %Parameters of each component (means us and variances vs)
        u1=0;v1=1; u2=4;v2=1; u3 =4; v3=1; %u3 will be multiplied for -1!!
        params=[u1 v1 u2 v2 u3 v3];
    %Mixing proportions
        mix=[.8 .1 .1];
    %proportion per component of mix mod
        N2=round(N.*mix);
    %actual data generation
        [data] = Generate_MixMod3CompData(datatype,N2, params);
        normalize=1; %standardize data for easier initialization
        if normalize==1
            data=(data-mean(data))/std(data);
        end 
    %real_noise_var=var(data(1:N2(1)));
    
%==========1-dim mixture model data generation ===============


 %========== MODEL FIT=============== 
    if options.initialization=='givens'
        %ms=[mean(data) mean(data)+2*std(data) -(mean(data)+2*std(data))]';
        ms=[0 3 -3]';vs=[1 1 1]';%ms=[0 2 -2]';         vs=[1 2 2]';
        options.MLinit=[ms vs]; 
        if options.K==3
        else
            options.inits=options.MLinit.*repmat(options.comp',1,2);
        end
    end
    tic
    mix=SIN_init_VB_MM(data,options);        
    src=SIN_VB_MixMod(data,mix)  ;
    time=toc;
    cost=src.it;
    fprintf('VB took %d seconds (%d iterations) \n',time,cost )
%========== MODEL FIT===============      




%========== Visualization===============      
figure(1);clf        
[f,x]=hist(data,50);
bar(x,f/trapz(x,f));
hold on
invgam=@(x,a,b) b^a/gamma(a).*(1./x).^(a+1).*exp(-b./x);
rage=-10:.001:10;
pos=find(rage>0);neg=find(rage<0);
if options.MM=='GIM'
    plt1=invgam(rage,src.shapes(1),src.scales(1));plt1(neg)=0;
    plt2=invgam(-rage, src.shapes(2),src.scales(2)   );plt2(pos)=0;
   title('GIM FIT')
else
   plt1=gampdf(rage,src.shapes(1),1/src.rates(1));plt1(neg)=0;
    plt2=gampdf(-rage,src.shapes(2),1/src.rates(2));plt2(pos)=0; 
    title('GGM FIT')
end
plot(rage,src.pi(1).*normpdf(rage,src.mu1,sqrt(1/src.tau1)),'r');hold on
plot(rage,src.pi(2).*plt1,'r');hold on
plot(rage,src.pi(3).*plt2,'r');hold on
        plot(rage,src.pi(1).*normpdf(rage,src.mu1,sqrt(1/src.tau1))+src.pi(2).*plt1 +src.pi(3).*plt2 ,'g');hold on

ylim=max(f/trapz(x,f));
set(gca,'ylim',[0 ylim])
axis tight
          
%========== Visualization===============      
