function [Train,Test] = divide_data2(ratings)
ratings = ratings./5;
[N,M]=size(ratings);
for j=1:M
    [I,J,Val] = find(ratings(:,j));
    %rand('state',0); %每次产生的随机数不会发生变化
    if (length(I)<5)
        [test,train] = crossvalind('LeaveMOut',nnz(ratings(:,j)),length(I));  %该命令返回交叉索引逻辑索引向量，
    else
        [test,train] = crossvalind('LeaveMOut',nnz(ratings(:,j)),5);
    end
        %从nnz(rating(:,j))个观察样本中随机选取ceil(nnz(rating(:,j))*sp个样本作为训练集，其余作为测试集。
    %值得注意的是，LeaveMOut在循环中使用不能保证产生的是互补集合，即每次循环的随机选取是独立的。
    %nnz函数返回(rating(:,j))中的非零元素的数目; ceil函数将nnz(rating(:,j))*sp)中的元素向上取整
    Train(:,j)=sparse(I(train), J(train), Val(train),N,1);
    Test(:,j)=sparse(I(test),J(test), Val(test), N,1);
end
end