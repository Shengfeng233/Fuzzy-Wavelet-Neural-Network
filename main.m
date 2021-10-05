%%%�������ӿ�
%%%Layout
%%%1.�����������ά�Ȳ����ڶ��㣨ģ��ǰ����
%%%2.����ģ��ǰ������������ģ������ȫ���У�
%%%3.�м䴦����FWNN���������
%%%4.���������������ɶ�Ӧ��С��������
%%%5.��ʧ����MSE
%%%6.����ѵ�����������ƫ����
%%%7.�ݶ��㷨(fminunc)

%���������ݼ�
load S.mat
load Y.mat
%ģ��ǰ������Ŀ�涨
pre=2;
feature_num=size(S,2);
m=pre^feature_num;
%��ʼ����������
[miu,sigma,b_mat,c_mat,w_mat]=Init_para(pre,feature_num,m);
%unroll��ʼ����������
x0=[miu(:);sigma(:);b_mat(:);c_mat(:);w_mat(:)];
%��ѭ��
fun=@(theta)costFunction(S,Y,pre,theta);
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton',...
    'MaxFunctionEvaluations',2*10^5	);
[x,fval,exitflag,output] = fminunc(fun,x0,options);

%unrollѵ�����������
miu=reshape(x(1:pre*feature_num),pre,feature_num);
sigma=reshape(x((pre*feature_num+1):2*pre*feature_num),pre,feature_num);
b_mat=reshape(x((2*pre*feature_num+1):(2*pre*feature_num+feature_num*m))...
    ,feature_num,m);
c_mat=reshape(x((2*pre*feature_num+feature_num*m+1):(2*pre*feature_num+2*feature_num*m))...
    ,feature_num,m);
w_mat=reshape(x((2*pre*feature_num+2*feature_num*m+1):(2*pre*feature_num+3*feature_num*m))...
    ,feature_num,m);
%����ѵ���Ĳ����������Ԥ��
fwmodel_out = Fwnn(S, Y, pre,miu,sigma,b_mat,c_mat,w_mat);
predict=fwmodel_out.Y_est;
%��ͼ�Ƚ�
q=linspace(1,100,100);
figure(1);
plot(q,Y);
hold on
plot(q,predict,'r--');
legend('True Value','FWNN Predict')