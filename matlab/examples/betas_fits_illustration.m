%generate data from a beta mixture model and fit it
clear
figure(1);clf
N_total=10000; %total number of samples

for sp=1:6
    subplot(2,3,sp)
    % parameters for generating synthetic data
    if sp==1
        mix=[.5 .5];  %mix proportions
        a=1;b=5; %beta parameters of components 1st and 2nd components
        a2=5;b2=1;
    elseif sp==2
        mix=[.9 .1];  a=1;b=5; a2=5;b2=1;
    elseif sp==3
        mix=[.1 .9];  a=1;b=5; a2=5;b2=1;   
    elseif sp==4
        mix=[.1 .9];  a=10;b=50; a2=5;b2=1;  
    elseif sp==5
        mix=[.9 .1];  a=10;b=50; a2=5;b2=1;  
    elseif sp==6
        mix=[.5 .5];  a=2;b=2; a2=2;b2=2;
    end

    %generate the data
    sample_sizes=round(N_total*mix);
    x1=betarnd(a,b,1,sample_sizes(1));
    x2=betarnd(a2,b2,1,sample_sizes(2));
    data=[x1 x2];

    %plot histogram of data
    range=0:1/1000:1;
    [f, y] = hist(data,30);
    %figure(1);clf
    bar(y, f / trapz(y, f),'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5); hold on
    plot(range,mix(1)*betapdf(range,a,b) + mix(2)*betapdf(range,a2,b2), 'r','Linewidth',2);



    % fit mixture models of 2 beta distributions
    MM=mmbetas_2comp(data);
    MM=mmbetas(data,2,0);

    %plot fit
    figure(1)
    plot(range,MM.pi(1)*betapdf(range,MM.beta_params(1,1),MM.beta_params(1,2)) + MM.pi(2)*betapdf(range,MM.beta_params(2,1),MM.beta_params(2,2)),'k','Linewidth',2);

end
