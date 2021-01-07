function[U,W,V,X,Y,Z,mu1,mu2,mu3,mu4,mu5,mu6] = BMMFinit(phi_x1, phi_x2, phi_x3, phi_x4, phi_x5, phi_x6, Xu_train, r, param)
%SOLVEDLCE 此处显示有关此函数的摘要
%   此处显示详细说明
%% random initializationcq
[row,col] = size(phi_x1); 
[~,colg] = size(phi_x2);
[~,colk] = size(phi_x3);
[~,colp] = size(phi_x4);
[~,colr] = size(phi_x5);
[~,colt] = size(phi_x6);

n_anchors = param.n_anchors;
lambda = param.lambda;
alpha = param.alpha;
beta= param.beta;
gamma = param.gamma;
omega = param.omega;
t = param.t;
maxItr = param.maxItr;

mu1 = 0.1;
mu2 = 0.2;
mu3 = 0.2;
mu4 = 0.2;
mu5 = 0.2;
mu6 = 0.1;

Phi_X = [sqrt(mu1)*phi_x1, sqrt(mu2)*phi_x2, sqrt(mu3)*phi_x3, sqrt(mu4)*phi_x4,sqrt(mu5)*phi_x5, sqrt(mu6)*phi_x6];

% constructing the adjacency matrix
A = construct_A(Phi_X, 1, true);


ST = Xu_train';
IDX = (Xu_train~=0);
IDXT = IDX';
maxS = max(max(Xu_train));
minS = min(Xu_train(Xu_train>0));

PT = Phi_X';
IDP = (Phi_X~=0);
IDPT = IDP';
maxP = max(max(Phi_X));
minP = min(min(Phi_X));

[n,m] = size(Xu_train);
[~,p] = size(Phi_X);

%rng(10)
U =  randn(r,m) ;
V =  randn(r,n) ;
W =  randn(r,p) ;
X = UpdateSVD(U);
Y = UpdateSVD(V);
Z = UpdateSVD(W);

if isfield(param,'maxItr')
    maxItr = param.maxItr;
else
    maxItr = 20;
end
if isfield(param,'tol')
    tol = param.tol;
else
    tol = 1e-5;
end
if isfield(param,'debug')
    debug = param.debug;
else
    debug = false;
end
converge = false;
it = 1;

        
if debug
    disp('Starting DLCEinit...');
    [loss,obj] = BMMFinitObj(maxS, maxP, minS, minP,Xu_train, Phi_X, ST, PT,IDX, IDP, A, U, W, V, X, Y, Z,alpha, beta,gamma,omega,lambda);
    disp(['loss value = ',num2str(loss)]);
    disp(['obj value = ',num2str(obj)]);
    
end

while ~converge
    U0 = U;
    V0 = V;
    W0 = W;
    X0 = X;
    Y0 = Y;
    Z0 = Z;
    
    for l = 1:p
        Vl = V(:,IDP(:,l)); 
        Pl = nonzeros(Phi_X(:,l));
        if isempty(Pl)
            continue;
        end
        Pl = ScaleScore(Pl,r,maxP,minP);
        Q = lambda*(Vl*Vl')+beta*length(Pl)*eye(r);%quadratic term
        L = lambda*Vl*Pl+2*beta*Z(:,l);% linear term
        W(:,l) = pinv(Q)*L;
    end
    parfor i = 1:m
        Vi = V(:,IDX(:,i));
        Si = nonzeros(Xu_train(:,i));
        if isempty(Si)
            continue;
        end
        Si = ScaleScore(Si,r,maxS,minS);
        Q = (1-lambda)*(Vi*Vi')+alpha*length(Si)*eye(r);%quadratic term
        L = (1-lambda)*Vi*Si+2*alpha*X(:,i);% linear term
        U(:,i) = pinv(Q)*L;
    end
    for j = 1:n
        Wj = W(:,IDPT(:,j)); 
        Uj = U(:,IDXT(:,j)); 
        Sj = nonzeros(ST(:,j));
        Pj = nonzeros(PT(:,j));
        if isempty(Sj)
            continue;
        end
        if isempty(Pj)
            continue;
        end
        Sj = ScaleScore(Sj,r,maxS,minS);
        Pj = ScaleScore(Pj,r,maxP,minP);
        Q = (1-lambda)*(Uj*Uj')+lambda*(Wj*Wj')+gamma*(length(Sj)+length(Pj))*eye(r);      
        L = (1-lambda)*Uj*Sj+lambda*Wj*Pj+omega*V*A(:,j)+2*gamma*Y(:,j);     
        V(:,j) = pinv(Q)*L;
    end
    W1 = W( :,1:col)./sqrt(mu1);
    W2 = W( :,col+1:2*col)./sqrt(mu2);
    W3 = W( :,2*col+1:3*col)./sqrt(mu3);
    W4 = W( :,3*col+1:4*col)./sqrt(mu4);
    W5 = W( :,4*col+1:5*col)./sqrt(mu5);
    W6 = W( :,5*col+1:end)./sqrt(mu6);
    h1 = sum(sum((phi_x1-V'*W1).^2)) ;
    h2 = sum(sum((phi_x2-V'*W2).^2)) ;
    h3 = sum(sum((phi_x3-V'*W3).^2)) ;
    h4 = sum(sum((phi_x4-V'*W4).^2)) ;
    h5 = sum(sum((phi_x5-V'*W5).^2)) ;
    h6 = sum(sum((phi_x6-V'*W6).^2)) ;
    mu1 = ((1/h1).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    mu2 = ((1/h2).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    mu3 = ((1/h3).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    mu4 = ((1/h4).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    mu5 = ((1/h5).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    mu6 = ((1/h6).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
   
    X = UpdateSVD(U);
    Y = UpdateSVD(V);
    Z = UpdateSVD(W);
    
    disp(['DLCEinit Iteration:',int2str(it)]);
    if it >= maxItr || max([norm(U-U0,'fro') norm(V-V0,'fro') norm(X-X0,'fro') norm(Y-Y0,'fro')]) < max([m n])*tol...
    || max([norm(V-V0,'fro') norm(W-W0,'fro') norm(Y-Y0,'fro') norm(Z-Z0,'fro')]) < max([p n])*tol
        converge = true;
    end

    if debug
        [loss,obj] = BMMFinitObj(maxS, maxP, minS, minP,Xu_train, Phi_X, ST, PT,IDX, IDP, A, U, W, V, X, Y, Z,alpha, beta,gamma,omega,lambda);
        disp(['loss value = ',num2str(loss)]);
        disp(['obj value = ',num2str(obj)]);
       
    end
    it = it+1;
end
end

function [loss, obj] = BMMFinitObj(maxS, maxP, minS, minP,Xu_train, Phi_X, ST, PT,IDX, IDP, A, B, C, D, X, Y, Z,alpha, beta,gamma,omega,lambda)
[n,m] = size(Xu_train);
[~,p] = size(Phi_X);
r = size(B,1);
loss1 = zeros(1,m);
loss2 = zeros(1,p);
for j = 1:m
    bj = B(:,j);
    Dj = D(:,IDX(:,j));
    DDj = Dj*Dj';
    term1 = bj'*DDj*bj;
    Sj = ScaleScore(nonzeros(Xu_train(:,j)),r,maxS,minS);
    term2 = 2*bj'*Dj*Sj;
    term3 = sum(Sj.^2);
    loss1(j) = term1-term2+term3;
end

for i = 1:p
    ci = C(:,i);
    Di= D(:,IDP(:,i));
    DDi = Di*Di';
    term1 = ci'*DDi*ci;
    Pi = ScaleScore(nonzeros(Phi_X(:,i)),r,maxP,minP);
    term2 = 2*ci'*Di*Pi;
    term3 = sum(Pi.^2);
    loss2(i) = term1-term2+term3;
end

E =sparse(diag(sum(A)));
ED = E * D';
AD = A * D';
reg = omega   * (tr(D', ED) - tr(D', AD));
 
for i = 1:m
    bi = B(:,i);
    Si = nonzeros(Xu_train(:,i));
    reg = reg+alpha*length(Si)*sum(bi.^2);
end

for k = 1:n
    dk = D(:,k);
    Pk = nonzeros(PT(:,k));
    Sk = nonzeros(ST(:,k));
    reg = reg+gamma*(length(Pk)+length(Sk))*sum(dk.^2);
end

for j = 1:p
    cj = C(:,j);
    Pj = nonzeros(Phi_X(:,j));
    reg = reg+beta*length(Pj)*sum(cj.^2);
end
loss = sum(loss1)+sum(loss2);

obj = loss+reg+alpha*norm(X-B,'fro').^2+beta*norm(C-Z,'fro').^2+gamma*norm(D-Y,'fro').^2;
end

function[trAB] = tr(A,B)
    trAB = sum(sum(A.*B));
end