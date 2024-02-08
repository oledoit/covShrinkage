function sigmahat=covMarket(Y,k)

% function sigmahat=covMarket(Y,k)
%
% Y (N*p): raw data matrix of N iid observations on p random variables
% sigmahat (p*p): invertible covariance matrix estimator
%
% Shrinks towards a one-factor market model, where the factor is defined 
%    as the cross-sectional average of all the random variables;
%    thanks to the idiosyncratic volatility of the residuals, the target 
%    preserves the diagonal of the sample covariance matrix 
%
% If the second (optional) parameter k is absent, not-a-number, or empty,
% then the algorithm demeans the data by default, and adjusts the effective
% sample size accordingly. If the user inputs k = 0, then no demeaning
% takes place; if (s)he inputs k = 1, then it signifies that the data Y has
% already been demeaned.
%
% This version: 01/2021, based on the 04/2014 version

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is released under the BSD 2-clause license.

% Copyright (c) 2014-2021, Olivier Ledoit and Michael Wolf
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% de-mean returns if required
[N,p]=size(Y);                      % sample size and matrix dimension
if (nargin<2)||isnan(k)||isempty(k) % default setting
   Y=Y-repmat(mean(Y),[N 1]);       % demean the raw data matrix
   k=1;                             % subtract one degree of freedom
end
n=N-k;                              % adjust effective sample size

% compute sample covariance matrix
sample=(Y'*Y)./n;

% compute shrinkage target
Ymkt=mean(Y,2); % equal-weighted market factor
covmkt=(Y'*Ymkt)./n; % covariance of original variables with common factor
varmkt=(Ymkt'*Ymkt)./n; % variance of common factor
target=covmkt*covmkt'./varmkt;
target(logical(eye(p)))=diag(sample);

% estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
Y2=Y.^2;
sample2=(Y2'*Y2)./n; % sample matrix of products of squared returns
piMat=sample2-sample.^2;
pihat=sum(sum(piMat));

% estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
gammahat=norm(sample-target,'fro')^2;

% diagonal part of the parameter that we call rho 
rho_diag=sum(diag(piMat));

% off-diagonal part of the parameter that we call rho 
temp=Y.*Ymkt(:,ones(1,p));
v1=(1/n)*Y2'*temp-covmkt(:,ones(1,p)).*sample;
roff1=sum(sum(v1.*covmkt(:,ones(1,p))'))/varmkt...
   -sum(diag(v1).*covmkt)/varmkt;
v3=(1/n)*temp'*temp-varmkt*sample;
roff3=sum(sum(v3.*(covmkt*covmkt')))/varmkt^2 ...
   -sum(diag(v3).*covmkt.^2)/varmkt^2;
rho_off=2*roff1-roff3;

% compute shrinkage intensity
rhohat=rho_diag+rho_off;
kappahat=(pihat-rhohat)/gammahat;
shrinkage=max(0,min(1,kappahat/n));

% compute shrinkage estimator
sigmahat=shrinkage*target+(1-shrinkage)*sample;

