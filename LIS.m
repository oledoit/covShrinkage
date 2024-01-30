function sigmahat=LIS(Y,k)          

% function sigmahat=LIS(Y,k) 
%
% Y (N*p): raw data matrix of N iid observations on p random variables
% sigmahat (p*p): invertible covariance matrix estimator
%
% Implements the linear-inverse shrinkage (LIS) estimator  
%    This is a nonlinear shrinkage estimator derived under on Stein's loss
%
% If the second (optional) parameter k is absent, not-a-number, or empty,
% then the algorithm demeans the data by default, and adjusts the effective
% sample size accordingly. If the user inputs k = 0, then no demeaning
% takes place; if (s)he inputs k = 1, then it signifies that the data Y has
% already been demeaned.
%
% This version: 01/2024

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is released under the BSD 2-clause license.

% Copyright (c) 2021, Olivier Ledoit and Michael Wolf
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

%%% EXTRACT sample eigenvalues sorted in ascending order and eigenvectors %%%
[N,p]=size(Y);                      % sample size and matrix dimension
if (nargin<2)||isnan(k)||isempty(k) % default setting
   Y=Y-repmat(mean(Y),[N 1]);       % demean the raw data matrix
   k=1;                             % subtract one degree of freedom
end
n=N-k;                              % adjust effective sample size
c=p/n;                              % concentration ratio
sample=(Y'*Y)./n;                   % sample covariance matrix
sample=(sample+sample')./2;         % make it even more symmetric numerically
[u,lambda]=eig(sample,'vector');    % spectral decomposition
u=real(u);                          % sample eigenvectors should be real
lambda=real(lambda);                % sample eigenvalues should also be real
[lambda,isort]=sort(lambda);        % sort eigenvalues in ascending order
u=u(:,isort);                       % eigenvectors follow their eigenvalues
%%% COMPUTE Linear-Inverse Shrinkage estimator of the covariance matrix %%%
h=min(c^2,1/c^2)^0.35/p^0.35;             % smoothing parameter
invlambda=1./lambda(max(1,p-n+1):p);      % inverse of (non-null) eigenvalues
Lj=repmat(invlambda,[1 min(p,n)])';       % like  1/lambda_j
Lj_i=Lj-Lj';                              % like (1/lambda_j)-(1/lambda_i)
theta=mean(Lj.*Lj_i./(Lj_i.^2+h^2.*Lj.^2),2);     % smoothed Stein shrinker
if p<=n % case where sample covariance matrix is not singular
   deltahat_1=(1-c)*invlambda+2*c*invlambda.*theta; % shrunk inverse eigenvalues
else % case where sample covariance matrix is singular
   error('p must be <= n for Stein''s loss')
end
deltaLIS_1=max(deltahat_1,min(invlambda));
sigmahat=u*diag(1./deltaLIS_1)*u';          % reconstruct covariance matrix
