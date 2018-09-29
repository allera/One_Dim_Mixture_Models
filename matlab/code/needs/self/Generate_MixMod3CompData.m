function [mixmoddata, labels] = MixMod3CompData(datatype,N, params)

%generate data from mix model of three compoentes, Gauss centered in zero,
%a positive and a negative activation. activation can be gamma or inverse
%gamma.

%datatype= 1,2 or 3: Gauss/Gauss, Gauss/Gammas, Gauss/Inverse Gammas
%N=[N1 N2 N3] number of samples per component

%params = [u1 v1 u2 v2 u3 v3] %u3 <0


invgam = @(x,a,b) b^a/gamma(a).*(1./x).^(a+1).*exp(-b./x);
%use gampdf%gam = @(x,a,b) (1/(b^a*gamma(a))).*(x).^(a-1).*exp(-x./b);
alphaIG=@(mu,var)(mu^2/var)+2;
betaIG=@(mu,var)(mu*((mu^2/var)+1));
alphaGm=@(mu,var)(mu^2/var);
betaGm=@(mu,var)(var/mu);




u1=params(1);u2=params(3);u3=params(5);
v1=params(2);v2=params(4);v3=params(6);

%generate noise distribution
data1= u1 + sqrt(v1).*randn(1,N(1));
%fprintf('Generating data ....\n')
    if datatype==2 % gamma activations

        g=gamrnd(alphaGm(u2,v2),betaGm(u2,v2),[1 N(2)]);
        data2=g;
        g2=gamrnd(alphaGm(u3,v3),betaGm(u3,v3),[1 N(3)]);
        data3=-g2;  
    elseif datatype==3 %inverse gamma activation

        g=gamrnd(alphaIG(u2,v2),1./betaIG(u2,v2),[1 N(2)]);
        data2=1./g;

        g2=gamrnd(alphaIG(u3,v3),1./betaIG(u3,v3),[1 N(3)]);
        data3=-1./g2;
    else %gauss activation
        data2= u2 + sqrt(v2).*randn(1,N(2));
        data3= -u3 + sqrt(v3).*randn(1,N(3));

    end

mixmoddata=[data1 data2 data3];
labels=[ones(1,numel(data1)) 2*ones(1,numel(data2)) 3*ones(1,numel(data3))];
%fprintf('done.\n')

end