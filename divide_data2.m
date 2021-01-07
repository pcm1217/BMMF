function [Train,Test] = divide_data2(ratings)
ratings = ratings./5;
[N,M]=size(ratings);
for j=1:M
    [I,J,Val] = find(ratings(:,j));
    %rand('state',0); %ÿ�β�������������ᷢ���仯
    if (length(I)<5)
        [test,train] = crossvalind('LeaveMOut',nnz(ratings(:,j)),length(I));  %������ؽ��������߼�����������
    else
        [test,train] = crossvalind('LeaveMOut',nnz(ratings(:,j)),5);
    end
        %��nnz(rating(:,j))���۲����������ѡȡceil(nnz(rating(:,j))*sp��������Ϊѵ������������Ϊ���Լ���
    %ֵ��ע����ǣ�LeaveMOut��ѭ����ʹ�ò��ܱ�֤�������ǻ������ϣ���ÿ��ѭ�������ѡȡ�Ƕ����ġ�
    %nnz��������(rating(:,j))�еķ���Ԫ�ص���Ŀ; ceil������nnz(rating(:,j))*sp)�е�Ԫ������ȡ��
    Train(:,j)=sparse(I(train), J(train), Val(train),N,1);
    Test(:,j)=sparse(I(test),J(test), Val(test), N,1);
end
end