clear all;
warning off;
load MovieLens-1M\MovieLens-1M\ratings.mat ;
load MovieLens-1M\MovieLens-1M\Actor.mat ;
load MovieLens-1M\MovieLens-1M\Genre.mat ;
load MovieLens-1M\MovieLens-1M\Keyword.mat ;
load MovieLens-1M\MovieLens-1M\Plot.mat ;
load MovieLens-1M\MovieLens-1M\Review.mat ;
load MovieLens-1M\MovieLens-1M\Tag.mat ;

[Xu_train,Xu_test]= divide_data2(ratings);

Xu_train =Xu_train';
Xu_test =Xu_test';
a =1;
r = 8; 
n_anchors = 300;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.debug = true;
param.Init = true;
param.maxItr = 20;
param.n_anchors = n_anchors;  %number of anchors
param.t = 6; %number of multi_modal feature
%%%%%%%%%%%%%%%%超参%%%%%%%%%%%%%%%%%%

param.lambda=0.9;%0.1
param.omega = 0.01;
param.alpha = 0.1;  %0.001
param.beta =0.1;
param.gamma = 0.1;
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 中心 核化
fprintf('centralizing data...\n');
        Ntrain = size(actor,1); 

% 取锚点
sample = randsample(Ntrain, param.n_anchors);
anchorA = actor(sample,:);
anchorG = genre(sample,:);
anchorK = keyword(sample,:);
anchorP = plot(sample,:);
anchorR = review(sample,:);
anchorT = tag(sample,:);

sigmaA = 0.2; 
sigmaG = 3; 
sigmaK = 0.3; 
sigmaP = 0.3; 
sigmaR = 0.3;
sigmaT = 7;

PhaA = exp(-sqdist(actor,anchorA)/(2*sigmaA*sigmaA));
PhaA = [PhaA, ones(Ntrain,1)];
PhgG = exp(-sqdist(genre,anchorG)/(2*sigmaG*sigmaG));
PhgG = [PhgG, ones(Ntrain,1)];
PhkK = exp(-sqdist(keyword,anchorK)/(2*sigmaK*sigmaK));
PhkK = [PhkK, ones(Ntrain,1)];
PhpP = exp(-sqdist(plot,anchorP)/(2*sigmaP*sigmaP));
PhpP = [PhpP, ones(Ntrain,1)];
PhrR = exp(-sqdist(review,anchorR)/(2*sigmaR*sigmaR));
PhrR = [PhrR, ones(Ntrain,1)];
PhtT = exp(-sqdist(tag,anchorT)/(2*sigmaT*sigmaT));
PhtT = [PhtT, ones(Ntrain,1)];

fprintf('run %d starts...\n', a);
   
A_temp = PhaA;
G_temp = PhgG;
K_temp = PhkK;
P_temp = PhpP;
R_temp = PhrR;
T_temp = PhtT;


[B,C,D,X,Y,Z,mu1,mu2,mu3,mu4,mu5,mu6] = BMMF(A_temp, G_temp, K_temp, P_temp, R_temp, T_temp, Xu_train, r, param);

fprintf('start evaluating...\n');
k=20;
ndcg= rating_metric(Xu_test, D', B', k);
fprintf('ndcg:%0.4f\n',ndcg);
        

