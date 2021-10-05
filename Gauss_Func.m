function A=Gauss_Func(x,miu,sigma)
pre_in=size(sigma,1);
x=repmat(x,pre_in,1);
% miu=repmat(miu,1,size(x,2));
% sigma=repmat(sigma,1,size(x,2));
A=exp(-1/2*((x-miu)./sigma).^2);
end