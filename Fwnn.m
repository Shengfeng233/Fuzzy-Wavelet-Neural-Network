function  [fwmodel] = Fwnn(S, Y, pre,miu,sigma,b_mat,c_mat,w_mat)
%%%Layout
%%%1.�����������ά�Ȳ����ڶ��㣨ģ��ǰ����
%%%2.����ģ��ǰ������������ģ������ȫ���У�
%%%3.�м䴦���������������
%%%4.���������������ɶ�Ӧ��С��������
%%%5.��ʧ����MSE
%%%6.����ѵ��������ƫ����
%%%7.�ݶ��㷨

%�������γ��
[sample_num,feature_num]=size(S);
%����������γ���Ƿ�ƥ��
[m_out,out_dim]=size(Y);
if(m_out~=sample_num)
    A=input("���ݲ�ƥ��");
end
if(out_dim~=1)
    B=input("�������һά����");
end
y_est=zeros(sample_num,1);
yita_all=zeros(sample_num,1);
yita=zeros(pre^feature_num,sample_num);
phai=zeros(pre^feature_num,sample_num);
index=zeros(pre^feature_num,feature_num);
for i=1:sample_num%ÿ�����ݰ����������ģ������
x=S(i,:);
%Layer2,����ģ��ǰ����,ÿһ�ж�Ӧһ��ǰ����
A_fuzz=Gauss_Func(x,miu,sigma);%A_fuzz(pre*feature_num)
%Layer3,����ģ������m��
m=pre^feature_num;
[fuzz_rule,index]=Fuzz_Rule(feature_num,pre,A_fuzz);%fuzz_rule=(pre^feature_num,1)
%Layer4,���ģ�������Ȩ��
yita_all(i)=sum(fuzz_rule(:));
yita(:,i)=fuzz_rule/yita_all(i);
%Layer5,��pre^feature_num����С��������
phai(:,i)=Wnn(x,feature_num,m,b_mat,c_mat,w_mat);%phai(m,1)
%Layer6,���
y_est(i)=(yita(:,i)')*phai(:,i);
end
fwmodel=struct('Y_est',y_est,'Y_true',Y,'X',S,'Miu_mat',miu, ...
    'Sigma_mat',sigma,'B_mat',b_mat,'C_mat',c_mat, ...
    'W_mat',w_mat,'Yita',yita,'Yita_all',yita_all,'Pre',pre, ...
    'Phai',phai,'Index',index);
end