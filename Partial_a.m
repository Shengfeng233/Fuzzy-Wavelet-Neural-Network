function p_A_inside=Partial_a(x,y,miu,sigma,feature_num,...
    pre,index,phai,yita_all)
    %从y对A1,i1偏导
    p_A_inside=zeros(pre,feature_num);
    A_fuzz=Gauss_Func(x,miu,sigma);
for i=1:feature_num
    A_fuzz_d=A_fuzz;
    A_fuzz_d(:,i)=[];
    index(:,[1,i])=index(:,[i,1]);
    for j=1:pre
    %对非第一列元素去除相应列，非第一行元素对应index值为j
    [sub_fuzz,k]=Fuzz_Rule(feature_num-1,pre,A_fuzz_d);
    k_add=[j*ones(size(sub_fuzz,1)),k];
    up_1=0;
    up_2=0;
    for i1=1:size(k_add,1)
        for i2=1:size(index,1)
            if(isequal(k_add(i1,:),index(i2,:)))
                up_1=up_1+sub_fuzz(i1)*phai(i2);
                up_2=up_2+y*sub_fuzz(i1);
            end 
        end
    end
    p_A_inside(i,j)=(up_1-up_2)/yita_all;
    end
    index(:,[1,i])=index(:,[i,1]);
end
end
