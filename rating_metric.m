function ndcg= rating_metric(test, P, Q, k)
[I,J,V] = find(test);  
pred_val = sum(P(I,:) .* Q(J,:), 2);  
all_col = [I,J,V,pred_val];
act_col = sortrows(all_col, [1,-3]);  
pred_col = sortrows(all_col,[1,-4]);  
item_count = full(sum(test>0,2));  
cum_item_count = cumsum(item_count);
cum_item_count = [0;cum_item_count];
num_items = size(test,1);
iind = 1;
ndcg_all = zeros(num_items - sum(sum(test>0)==0),k);
for i=1:num_items
    if item_count(i) == 0
        continue;
    end
    i_start = cum_item_count(i)+1;
    i_end = cum_item_count(i+1);
    act = act_col(i_start:i_end,3);
    discount = log2((1:k)'+1);
    pred = pred_col(i_start:i_end,3);
    if k > length(act)
        act_extend = [act; zeros(k-length(act),1)];
        pred_extend = [pred; zeros(k-length(act),1)];
    else
        act_extend = act(1:k);
        pred_extend = pred(1:k);
    end
    idcg = cumsum((2.^act_extend - 1) ./ discount);
    dcg = cumsum((2.^pred_extend - 1) ./discount);
    ndcg_all(iind,:) = dcg ./ idcg;
    iind = iind + 1;
end
ndcg = mean(ndcg_all);
end