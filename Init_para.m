function [miu,sigma,b_mat,c_mat,w_mat]=Init_para(pre,feature_num,m)
%暂用置1初始化，也可采用c-clusteting
miu=ones(pre,feature_num);
sigma=ones(pre,feature_num);
b_mat=rand(feature_num,m);
c_mat=rand(feature_num,m);
w_mat=rand(feature_num,m);
end