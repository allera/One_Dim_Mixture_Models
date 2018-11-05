%generate data from a beta mixture model and fit it
clear

%add code to path
addpath(genpath('../code/MixMod/'))
addpath(genpath('../code/needs/'))

% parameters for generating synthetic data
N_total=10000; %total number of samples
mix=[.1 .9];  %mix proportions
a=1;b=5; %beta parameters of components 1 nad 2
a2=5;b2=1;


%generate the data
sample_sizes=round(N_total*mix);
x1=betarnd(a,b,1,sample_sizes(1));
x2=betarnd(a2,b2,1,sample_sizes(2));
data=[x1 x2];

%plot histogram of data
range=0:1/1000:1;
[f, y] = hist(data,30);
figure(1);clf
bar(y, f / trapz(y, f),'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5); hold on
plot(range,mix(1)*betapdf(range,a,b) + mix(2)*betapdf(range,a2,b2), 'r','Linewidth',2);


if 0  % fit mixture models of 2 beta distributions
    MM=mmbetas_2comp(data);
    %plot fit
    plot(range,MM.pi(1)*betapdf(range,MM.beta_params(1,1),MM.beta_params(1,2)) + MM.pi(2)*betapdf(range,MM.beta_params(2,1),MM.beta_params(2,2)),'g','Linewidth',2);
end

if 1
    graphical=0;
    number_of_components=2;
    MM=mmbetas(data,number_of_components,graphical);
    estimated_parameters=MM.beta_params;
    estimatedPI=MM.pi;
    for i=1:number_of_components
        plot(range,estimatedPI(i)*betapdf(range,estimated_parameters(i,1),estimated_parameters(i,2)), 'r--','Linewidth',1);
    end    
    mix_pdf=estimatedPI(1)*betapdf(range,estimated_parameters(1,1),estimated_parameters(1,2));
    for i=2:number_of_components
       mix_pdf=mix_pdf+ estimatedPI(i)*betapdf(range,estimated_parameters(i,1),estimated_parameters(i,2));
    end
    plot(range,mix_pdf, 'r','Linewidth',2);
    %pause(1)
end



% %Use for estimate dimesnionality of PCA?
% K=20; %Number hidden sources
% T=180; % Number time points
% N=10000; % Number voxels
% randn('seed', 1);
% rand('seed', 1);
% X=AS+E
% S=randn(K,T);%Sources
% S=gamrnd(5,25,K,T);
% A=randn(N,K);
% E=10*randn(N,T); % add noise?
% X=(A*S)+E;
% [W,D]=eig(cov(X));
% lambdas=sort(diag(D),'descend');
% st_lambdas=lambdas/max(lambdas);
% figure(2);clf;
% subplot(1,2,1);
% plot(st_lambdas)
% subplot(1,2,2);
% plot histogram of data
% [f, y] = hist(st_lambdas,50);
% bar(y, f / trapz(y, f),'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5); hold on
%     
% MM=mmbetas(st_lambdas',2,0);
% estimated_parameters=MM.beta_params;
% estimatedPI=MM.pi;
% 
% plot(range,estimatedPI(1)*betapdf(range,estimated_parameters(1,1),estimated_parameters(1,2)) + estimatedPI(2)*betapdf(range,estimated_parameters(2,1),estimated_parameters(2,2)), 'r','Linewidth',2);
% 


% 
% 
