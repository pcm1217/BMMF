function[B,C,D,X,Y,Z,mu1,mu2,mu3,mu4,mu5,mu6] = BMMF(phi_x1, phi_x2, phi_x3, phi_x4, phi_x5, phi_x6, Xu_train, r, param)
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
Init = param.Init;
lambda = param.lambda;
alpha = param.alpha;
beta= param.beta;
gamma = param.gamma;
omega = param.omega;
t = param.t;
converge = false;
it = 1;
maxItr2 = 5;


if Init
    if (isfield(param,'B0') &&  isfield(param,'C0')&&  isfield(param,'D0') && isfield(param,'X0') && isfield(param,'Y0')) &&  isfield(param,'Z0')
        B0 = param.B0; C0 = param.C0; D0 = param.D0; X0 = param.X0; Y0 = param.Y0; Z0 = param.Z0;
    else
        [U,W,V,X0,Y0,Z0,mu1,mu2,mu3,mu4,mu5,mu6] = BMMFinit(phi_x1, phi_x2, phi_x3, phi_x4, phi_x5, phi_x6, Xu_train, r, param);
        B0 = sign(U); B0(B0 == 0) = 1; 
        C0 = sign(W);C0(C0 == 0) = 1;
        D0 = sign(V); D0(D0 == 0) = 1;  
    end
end

Phi_X = [sqrt(mu1)*phi_x1, sqrt(mu2)*phi_x2, sqrt(mu3)*phi_x3, sqrt(mu4)*phi_x4,sqrt(mu5)*phi_x5, sqrt(mu6)*phi_x6];
% constructing the adjacency matrix
A = construct_A(Phi_X, 1, true);

[n,m] = size(Xu_train);
[~,p] = size(Phi_X);
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

B1 = B0;
C1 = C0;
D1 = D0;
B = B0;
C = C0;
D = D0;
X = X0;
Y = Y0;
Z = Z0;

if debug
        [loss,obj] = BMMFobj(maxS, maxP, minS, minP,Xu_train, Phi_X, ST, PT,IDX, IDP, A, B, C, D, X, Y, Z,alpha, beta,gamma,omega,lambda);
        disp('Starting BMMF.....')
        disp(['loss value = ',num2str(loss)]);
        disp(['obj value = ',num2str(obj)]);
end

while ~converge
    B0 = B;
    D0 = D;
    C0 = C;
    B = B0;
    C = C0;
    D = D0;
    X0 = X;
    Y0 = Y;
    Z0 = Z;
                  
    for i = 1:m
        d = D(:,IDX(:,i));
        b = B(:,i);
        DCDmex(b,(1-lambda)*(d*d'),(1-lambda)*d*ScaleScore(nonzeros(Xu_train(:,i)),r,maxS,minS), alpha*X(:,i),maxItr2);
        B(:,i) = b;
    end
    
    for l = 1:p
        d = D(:,IDP(:,l));
        c = C(:,l);
        DCDmex(c,lambda*(d*d'),lambda*d*ScaleScore(nonzeros(Phi_X(:,l)),r,maxP,minP), beta*Z(:,l),maxItr2);
        C(:,l) = c;
    end
    
    for  j = 1:n
        b = B(:,IDXT(:,j));
        c = C(:,IDPT(:,j));
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
        Q = (1-lambda)*(b*b')+lambda*(c*c')+gamma*(length(Sj)+length(Pj))*eye(r);
        d = D(:,j);
        DCDmex(d,Q,(1-lambda)*b*Sj+lambda*c*Pj, omega*D*A(:,j)+2*gamma*Y(:,j),maxItr2);
        D(:,j) = d;
    end   
    
    C1 = C( :,1:col)./sqrt(mu1);
    C2 = C( :,col+1:2*col)./sqrt(mu2);
    C3 = C( :,2*col+1:3*col)./sqrt(mu3);
    C4 = C( :,3*col+1:4*col)./sqrt(mu4);
    C5 = C( :,4*col+1:5*col)./sqrt(mu5);
    C6 = C( :,5*col+1:end)./sqrt(mu6);
    h1 = sum(sum((phi_x1-D'*C1).^2)) ;
    h2 = sum(sum((phi_x2-D'*C2).^2)) ;
    h3 = sum(sum((phi_x3-D'*C3).^2)) ;
    h4 = sum(sum((phi_x4-D'*C4).^2)) ;
    h5 = sum(sum((phi_x5-D'*C5).^2)) ;
    h6 = sum(sum((phi_x6-D'*C6).^2)) ;
    mu1 = ((1/h1).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    mu2 = ((1/h2).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    mu3 = ((1/h3).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    mu4 = ((1/h4).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    mu5 = ((1/h5).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    mu6 = ((1/h6).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1))+(1/h3).^(1/(t-1))+(1/h4).^(1/(t-1))+(1/h5).^(1/(t-1))+(1/h6).^(1/(t-1)));
    
    X = UpdateSVD(B);
    Y = UpdateSVD(D);
    Z = UpdateSVD(C);
    
    if debug
        [loss,obj] = BMMFobj(maxS, maxP, minS, minP,Xu_train, Phi_X, ST, PT,IDX, IDP, A, B, C, D, X, Y, Z,alpha, beta,gamma,omega,lambda);
        disp(['loss value = ',num2str(loss)]);
        disp(['obj value = ',num2str(obj)]);
    end
    disp(['DLCE at bit ',int2str(r),' Iteration:',int2str(it)]);
        
    if it >= maxItr || (sum(sum(B~=B0)) == 0 && sum(sum(D~=D0)) == 0 && sum(sum(C~=C0)) == 0)
        converge = true;
    end
    it =it+1;
end
end

function [loss, obj] = BMMFobj(maxS, maxP, minS, minP,Xu_train, Phi_X, ST, PT,IDX, IDP, A, B, C, D, X, Y, Z,alpha, beta,gamma,omega,lambda)
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
 
loss = sum(loss1)+sum(loss2);
obj = loss+reg-2*alpha*trace(B*X')-2*beta*trace(C*Z')-2*gamma*trace(D*Y');
end

function[trAB] = tr(A,B)
    trAB = sum(sum(A.*B));
end