function sigmahat=cov2Para(Y,k)

% function sigmahat=cov2Para(Y,k)
%
% Y (N*p): raw data matrix of N iid observations on p random variables
% sigmahat (p*p): invertible covariance matrix estimator
%
% Shrinks towards two-parameter matrix:
%    all variances of the target are the same as one another
%    all covariances of the target are the same as one another
%
% If the second (optional) parameter k is absent, not-a-number, or empty,
% then the algorithm demeans the data by default, and adjusts the effective
% sample size accordingly. If the user inputs k = 0, then no demeaning
% takes place; if (s)he inputs k = 1, then it signifies that the data Y has
% already been demeaned.
%
% This version: 01/2021, based on the 06/2009 version

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is released under the BSD 2-clause license.

% Copyright (c) 2009-2021, Olivier Ledoit and Michael Wolf
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
meanvar=mean(diag(sample));
meancov=sum(sum(sample(~eye(p))))/p/(p-1);
target=meanvar*eye(p)+meancov*(~eye(p));

% estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
Y2=Y.^2;
sample2=(Y2'*Y2)./n; % sample covariance matrix of squared returns
piMat=sample2-sample.^2;
pihat=sum(sum(piMat));

% estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
gammahat=norm(sample-target,'fro')^2;

% diagonal part of the parameter that we call rho 
rho_diag=sum(sum(sample2))/p-trace(sample)^2/p;

% off-diagonal part of the parameter that we call rho 
sum1=sum(Y,2);
sum2=sum(Y2,2);
rho_off1=sum((sum1.^2-sum2).^2)/p/n;
rho_off2=(sum(sum(sample))-trace(sample))^2/p;
rho_off=(rho_off1-rho_off2)/(p-1);

% compute shrinkage intensity
rhohat=rho_diag+rho_off;
kappahat=(pihat-rhohat)/gammahat;
shrinkage=max(0,min(1,kappahat/n));

% compute shrinkage estimator
sigmahat=shrinkage*target+(1-shrinkage)*sample;

