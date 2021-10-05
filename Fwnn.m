function  [fwmodel] = Fwnn(S, Y, pre,miu,sigma,b_mat,c_mat,w_mat)
%%%Layout
%%%1.根据输入变量维度产生第二层（模糊前规则）
%%%2.根据模糊前规则的组合生成模糊规则（全排列）
%%%3.中间处理，在主函数中完成
%%%4.根据输入数据生成对应的小波神经网络
%%%5.损失函数MSE
%%%6.各个训练参数的偏导数
%%%7.梯度算法

%输入矩阵纬度
[sample_num,feature_num]=size(S);
%检测输出矩阵纬度是否匹配
[m_out,out_dim]=size(Y);
if(m_out~=sample_num)
    A=input("数据不匹配");
end
if(out_dim~=1)
    B=input("输出仅限一维数据");
end
y_est=zeros(sample_num,1);
yita_all=zeros(sample_num,1);
yita=zeros(pre^feature_num,sample_num);
phai=zeros(pre^feature_num,sample_num);
index=zeros(pre^feature_num,feature_num);
for i=1:sample_num%每个数据挨个输入产生模糊网络
x=S(i,:);
%Layer2,产生模糊前规则,每一列对应一组前规则
A_fuzz=Gauss_Func(x,miu,sigma);%A_fuzz(pre*feature_num)
%Layer3,产生模糊规则共m条
m=pre^feature_num;
[fuzz_rule,index]=Fuzz_Rule(feature_num,pre,A_fuzz);%fuzz_rule=(pre^feature_num,1)
%Layer4,输出模糊网络的权重
yita_all(i)=sum(fuzz_rule(:));
yita(:,i)=fuzz_rule/yita_all(i);
%Layer5,共pre^feature_num个子小波神经网络
phai(:,i)=Wnn(x,feature_num,m,b_mat,c_mat,w_mat);%phai(m,1)
%Layer6,输出
y_est(i)=(yita(:,i)')*phai(:,i);
end
fwmodel=struct('Y_est',y_est,'Y_true',Y,'X',S,'Miu_mat',miu, ...
    'Sigma_mat',sigma,'B_mat',b_mat,'C_mat',c_mat, ...
    'W_mat',w_mat,'Yita',yita,'Yita_all',yita_all,'Pre',pre, ...
    'Phai',phai,'Index',index);
end