%%%主函数接口
%%%Layout
%%%1.根据输入变量维度产生第二层（模糊前规则）
%%%2.根据模糊前规则的组合生成模糊规则（全排列）
%%%3.中间处理，在FWNN函数中完成
%%%4.根据输入数据生成对应的小波神经网络
%%%5.损失函数MSE
%%%6.各个训练参数矩阵的偏导数
%%%7.梯度算法(fminunc)

%测试用数据集
load S.mat
load Y.mat
%模糊前规则数目规定
pre=2;
feature_num=size(S,2);
m=pre^feature_num;
%初始化参数矩阵
[miu,sigma,b_mat,c_mat,w_mat]=Init_para(pre,feature_num,m);
%unroll初始化参数矩阵
x0=[miu(:);sigma(:);b_mat(:);c_mat(:);w_mat(:)];
%主循环
fun=@(theta)costFunction(S,Y,pre,theta);
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton',...
    'MaxFunctionEvaluations',2*10^5	);
[x,fval,exitflag,output] = fminunc(fun,x0,options);

%unroll训练后参数矩阵
miu=reshape(x(1:pre*feature_num),pre,feature_num);
sigma=reshape(x((pre*feature_num+1):2*pre*feature_num),pre,feature_num);
b_mat=reshape(x((2*pre*feature_num+1):(2*pre*feature_num+feature_num*m))...
    ,feature_num,m);
c_mat=reshape(x((2*pre*feature_num+feature_num*m+1):(2*pre*feature_num+2*feature_num*m))...
    ,feature_num,m);
w_mat=reshape(x((2*pre*feature_num+2*feature_num*m+1):(2*pre*feature_num+3*feature_num*m))...
    ,feature_num,m);
%根据训练的参数矩阵进行预测
fwmodel_out = Fwnn(S, Y, pre,miu,sigma,b_mat,c_mat,w_mat);
predict=fwmodel_out.Y_est;
%作图比较
q=linspace(1,100,100);
figure(1);
plot(q,Y);
hold on
plot(q,predict,'r--');
legend('True Value','FWNN Predict')