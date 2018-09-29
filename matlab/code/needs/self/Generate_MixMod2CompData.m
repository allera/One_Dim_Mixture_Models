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



u=[params(1) params(3)];
v=[params(2) params(4)];

u1=params(1);u2=params(3);
v1=params(2);v2=params(4);

for i=1:2 %each component can be gauss gamma or inv gamma
    if datatype(i)==1
        data{i}=u(i) + sqrt(v(i)).*randn(1,N(i));

    elseif datatype(i)==2 % gamma activations
        g=gamrnd(alphaGm(u(i),v(i)),betaGm(u(i),v(i)),[1 N(i)]);
        data{i}=g; 
        
    elseif datatype(i)==3 %inverse gamma activation

        g=gamrnd(alphaIG(u(i),v(i)),1./betaIG(u(i),v(i)),[1 N(i)]);
        data{i}=1./g;

    end
end

mixmoddata=[data{1} data{2}];
labels=[ones(1,numel(data{1})) 2*ones(1,numel(data{2}))];
%fprintf('done.\n')

end