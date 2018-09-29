function [ P ] = inversegamcdf( X,A,B )
%inversegamcdf Inverse gamma cumulative distribution function.
%   Y = inversegamcdf(X,A,B) returns the inverse gamma cumulative
%   distribution function with shape and scale parameters A and B,
%   respectively, at the values in X. The size of P is the common size of
%   the input arguments. A scalar input functions is a constant matrix of
%   the same size as the other inputs.

P = gammainc(B./X,A,'upper');

%http://csdspnest.blogspot.nl/2014/03/compute-inverse-gamma-pdf-and-cdf-in.html
end