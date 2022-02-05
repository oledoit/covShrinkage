function sigmahat=QIS(Y,k)        

% function sigmahat=QIS(Y,k) 
%
% Y (N*p): raw data matrix of N iid observations on p random variables
% sigmahat (p*p): invertible covariance matrix estimator
%
% Implements the quadratic-inverse shrinkage (QIS) estimator  
%    This is a nonlinear shrinkage estimator derived under the Frobenius loss
%    and its two cousins, Inverse Stein's loss and Mininum Variance loss
%
% If the second (optional) parameter k is absent, not-a-number, or empty,
% then the algorithm demeans the data by default, and adjusts the effective
% sample size accordingly. If the user inputs k = 0, then no demeaning
% takes place; if (s)he inputs k = 1, then it signifies that the data Y has
% already been demeaned.
%
% This version: 01/2021

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
[u,lambda]=eig(sample,'vector');    % spectral decomposition
[lambda,isort]=sort(lambda);        % sort eigenvalues in ascending order
u=u(:,isort);                       % eigenvectors follow their eigenvalues
%%% COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix %%%
h=min(c^2,1/c^2)^0.35/p^0.35;             % smoothing parameter
invlambda=1./lambda(max(1,p-n+1):p);      % inverse of (non-null) eigenvalues
Lj=repmat(invlambda,[1 min(p,n)])';       % like  1/lambda_j
Lj_i=Lj-Lj';                              % like (1/lambda_j)-(1/lambda_i)
theta=mean(Lj.*Lj_i./(Lj_i.^2+h^2.*Lj.^2),2);     % smoothed Stein shrinker
Htheta=mean(Lj.*(h.*Lj)./(Lj_i.^2+h^2.*Lj.^2),2); % its conjugate
Atheta2=theta.^2+Htheta.^2;                       % its squared amplitude
if p<=n % case where sample covariance matrix is not singular
   delta=1./((1-c)^2*invlambda+2*c*(1-c)*invlambda.*theta ...
      +c^2*invlambda.*Atheta2);           % optimally shrunk eigenvalues
else % case where sample covariance matrix is singular
   delta0=1./((c-1)*mean(invlambda));     % shrinkage of null eigenvalues
   delta=[repmat(delta0,[p-n 1]);1./(invlambda.*Atheta2)];
end
deltaQIS=delta.*(sum(lambda)/sum(delta)); % preserve trace
sigmahat=u*diag(deltaQIS)*u';             % reconstruct covariance matrix