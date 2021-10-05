function [fuzz_rule,k]=Fuzz_Rule(feature_num,pre,A_fuzz)
m=pre^feature_num;
fuzz_rule=ones(m,1);
k=ones(m,feature_num);%初始化index矩阵
origin_k=ones(1,feature_num);
for ik=1:m%生成index矩阵
    add_k=zeros(1,feature_num);
    total=ik-1;
    for ik2=1:feature_num
        add_k(feature_num-ik2+1)=mod(total,pre);
        total=(total-mod(total,pre))/pre;
    end
    k(ik,:)=origin_k+add_k;
end
for i1=1:m
    for i2=1:feature_num
    fuzz_rule(i1)=fuzz_rule(i1)*A_fuzz(k(i1,i2),i2);
    end
end
end