function[jval,gradientVec]=costFunction(S,Y,pre,theta)
%[miu,sigma,b_mat,c_mat,w_mat]
feature_num=size(S,2);
m=pre^feature_num;
miu=reshape(theta(1:pre*feature_num),pre,feature_num);
sigma=reshape(theta((pre*feature_num+1):2*pre*feature_num),pre,feature_num);
b_mat=reshape(theta((2*pre*feature_num+1):(2*pre*feature_num+feature_num*m))...
    ,feature_num,m);
c_mat=reshape(theta((2*pre*feature_num+feature_num*m+1):(2*pre*feature_num+2*feature_num*m))...
    ,feature_num,m);
w_mat=reshape(theta((2*pre*feature_num+2*feature_num*m+1):(2*pre*feature_num+3*feature_num*m))...
    ,feature_num,m);
[fwmodel] = Fwnn(S, Y, pre,miu,sigma,b_mat,c_mat,w_mat);
jval=1/size(fwmodel.X,1)*sum((fwmodel.Y_true-fwmodel.Y_est).^2);
if nargout>1
    gradientVec=Gradient_form(fmodel);
end
end