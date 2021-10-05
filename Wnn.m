function phai=Wnn(x,feature_num,m,b_mat,c_mat,w_mat)
x=repmat(x',1,m);
mid=((x-b_mat)./c_mat).^2;
phai=sum(w_mat.*((ones(feature_num,m)-mid).*(exp(-1/2*mid))),1);
phai=phai';
end