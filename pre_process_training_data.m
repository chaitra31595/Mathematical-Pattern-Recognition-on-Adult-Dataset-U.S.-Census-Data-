load('adult_train.mat')
[data_size features]=size(adult);
data_matrix(:,1)=adult{:,1};   % Age
data_matrix(:,10)=adult{:,3};  % fnlwt
data_matrix(:,11)=adult{:,5};  % education-num
data_matrix(:,46)=adult{:,11}; % capital-gain
data_matrix(:,47)=adult{:,12}; % capital-loss
data_matrix(:,48)=adult{:,13}; % hours-per-week

nation=[" Cambodia" " Canada" " China" " Columbia" " Cuba" " Dominican-Republic" " Ecuador" " El-Salvador" " England" " France" " Germany" " Greece" " Guatemala" " Haiti" " Honduras" " Hong" " Hungary" " India" " Iran" " Ireland" " Italy" " Jamaica" " Japan" " Laos" " Mexico" " Nicaragua" " Outlying-US(Guam-USVI-etc)" " Peru" " Philippines" " Poland" " Portugal" " Puerto-Rico" " Scotland" " South" " Taiwan" " Thailand" " Trinadad&Tobago" " United-States" " Vietnam" " Yugoslavia"];
workclass=[" Private" " Self-emp-not-inc" " Self-emp-inc" " Federal-gov" " Local-gov" " State-gov" " Without-pay" " Never-worked"];
marital_status=[" Married-civ-spouse" " Divorced" " Never-married" " Separated" " Widowed" " Married-spouse-absent" " Married-AF-spouse"];
occupation=[" Tech-support" " Craft-repair" " Other-service" " Sales" " Exec-managerial" " Prof-specialty" " Handlers-cleaners" " Machine-op-inspct" " Adm-clerical" " Farming-fishing" " Transport-moving" " Priv-house-serv" " Protective-serv" " Armed-Forces"];
relationship=[" Wife" " Own-child" " Husband" " Not-in-family" " Other-relative" " Unmarried"];
race=[" White" " Asian-Pac-Islander" " Amer-Indian-Eskimo" " Other" " Black"]; 
sex=[ " Female" " Male"];
k=1;l=1;m=1;n=1;o=1;p=1;q=1;
for i=1:data_size
    index1 = find(contains(workclass,adult{i,2}));
    if(~isempty(index1))
    data_matrix(i,1+index1)=1;
    else
       missing_index(1,k)=i;
       k=k+1;
    end

    index2 = find(contains(marital_status,adult{i,6}));
    if(~isempty(index2))
    data_matrix(i,11+index2)=1; 
    else
       missing_index(2,l)=i;
       l=l+1;
    end

    index3 = find(contains(occupation,adult{i,7}));
    if(~isempty(index3))
    data_matrix(i,18+index3)=1;
    else
       missing_index(3,m)=i;
       m=m+1;
    end
        
    index4 = find(contains(relationship,adult{i,8}));
    if(~isempty(index4))
    data_matrix(i,32+index4)=1;
    else
       missing_index(4,n)=i;
       n=n+1;
    end    

     index5 = find(contains(race,adult{i,9}));
    if(~isempty(index5))
    data_matrix(i,38+index5)=1;
    else
       missing_index(5,o)=i;
       o=o+1;
    end   

     index6 = find(contains(sex,adult{i,10}));
    if(~isempty(index6))
    data_matrix(i,43+index6)=1;
    else
       missing_index(6,p)=i;
       p=p+1;
    end
    
    index7 = find(contains(nation,adult{i,14}));
    if(~isempty(index7))
    data_matrix(i,48+index7)=1;
    else
       missing_index(7,q)=i;
       q=q+1;
    end
        
    if(strcmp(adult{i,15}," <=50K"))
       data_matrix(i,89)=-1;
    else
       data_matrix(i,89)=1;
    end
end


if(k>1)  %identifying missing values for workclass
for r=1:k-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(1,:)==i)) && (i~=missing_index(1,r)))
        dist1(r,j)=sqrt(sum((data_matrix(missing_index(1,r),1)-data_matrix(i,1)).^2+(data_matrix(missing_index(1,r),10:end)-data_matrix(i,10:end)).^2));
    else
        dist1(r,j)=Inf;
    end
    j=j+1;
end
[min_value,ind]=min(dist1(r,:));
in = find(contains(workclass,adult{ind,2}));
data_matrix(missing_index(1,r),1+in)=1;
end
end

if(l>1)  %identifying missing values for marital_status
for r=1:l-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(2,:)==i)) && (i~=missing_index(2,r)))
        dist2(r,j)=sqrt(sum((data_matrix(missing_index(2,r),1:11)-data_matrix(i,1:11)).^2)+sum((data_matrix(missing_index(2,r),19:end)-data_matrix(i,19:end)).^2));
    else
        dist2(r,j)=Inf;
    end
    j=j+1;     
end
[min_value,ind]=min(dist2(r,:));
in = find(contains(marital_status,adult{ind,6}));
data_matrix(missing_index(2,r),11+in)=1;
end
end
if(m>1)  %identifying missing values for occupation
for r=1:m-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(3,:)==i)) && (i~=missing_index(3,r)))
        dist3(r,j)=sqrt(sum((data_matrix(missing_index(3,r),1:18)-data_matrix(i,1:18)).^2)+sum((data_matrix(missing_index(3,r),33:end)-data_matrix(i,33:end)).^2));
    else
        dist3(r,j)=Inf;
    end
    j=j+1;     
end
[min_value,ind]=min(dist3(r,:));
in = find(contains(occupation,adult{ind,7}));
data_matrix(missing_index(3,r),18+in)=1;
end
end
if(n>1)  %identifying missing values for relationship
for r=1:n-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(4,:)==i)) && (i~=missing_index(4,r)))
        dist4(r,j)=sqrt(sum((data_matrix(missing_index(4,r),1:32)-data_matrix(i,1:32)).^2)+sum((data_matrix(missing_index(4,r),39:end)-data_matrix(i,39:end)).^2));
    else
        dist4(r,j)=Inf;
    end
    j=j+1;     
end
[min_value,ind]=min(dist4(r,:));
in = find(contains(relationship,adult{ind,8}));
data_matrix(missing_index(4,r),32+in)=1;
end
end
if(o>1)  %identifying missing values for race
for r=1:o-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(5,:)==i)) && (i~=missing_index(5,r)))
        dist5(r,j)=sqrt(sum((data_matrix(missing_index(5,r),1:38)-data_matrix(i,1:38)).^2)+sum((data_matrix(missing_index(5,r),44:end)-data_matrix(i,44:end)).^2));
    else
        dist5(r,j)=Inf;
    end
    j=j+1;     
end
[min_value,ind]=min(dist5(r,:));
in = find(contains(race,adult{ind,9}));
data_matrix(missing_index(5,r),38+in)=1;
end
end
if(p>1)  %identifying missing values for sex
for r=1:p-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(6,:)==i)) && (i~=missing_index(6,r)))
        dist6(r,j)=sqrt(sum((data_matrix(missing_index(6,r),1:43)-data_matrix(i,1:43)).^2)+sum((data_matrix(missing_index(6,r),46:end)-data_matrix(i,46:end)).^2));
    else
        dist6(r,j)=Inf;
    end    
    j=j+1;     
end
[min_value,ind]=min(dist6(r,:));
in = find(contains(sex,adult{ind,10}));
data_matrix(missing_index(6,r),43+in)=1;
end
end
if(q>1)  %identifying missing values for nation
for r=1:q-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(7,:)==i)) && (i~=missing_index(7,r)))
        dist7(r,j)=sqrt(sum((data_matrix(missing_index(7,r),1:48)-data_matrix(i,1:48)).^2)+sum((data_matrix(missing_index(7,r),end)-data_matrix(i,end)).^2));
    else
        dist7(r,j)=Inf;
    end
    j=j+1;
end
[min_value,ind]=min(dist7(r,:));
in = find(contains(nation,adult{ind,14}));
data_matrix(missing_index(7,r),48+in)=1;
end
end

data_matrix_train=data_matrix;
save('data_matrix_train.mat','data_matrix_train');

data_matrix_train_standardized=data_matrix_train;
data_matrix_train_standardized(:,1)=(data_matrix_train(:,1)-mean(data_matrix_train(:,1)))./(std(data_matrix_train(:,1),1));
data_matrix_train_standardized(:,10)=(data_matrix_train(:,10)-mean(data_matrix_train(:,10)))./(std(data_matrix_train(:,10),1));
data_matrix_train_standardized(:,11)=(data_matrix_train(:,11)-mean(data_matrix_train(:,11)))./(std(data_matrix_train(:,11),1));
data_matrix_train_standardized(:,46)=(data_matrix_train(:,46)-mean(data_matrix_train(:,46)))./(std(data_matrix_train(:,46),1));
data_matrix_train_standardized(:,47)=(data_matrix_train(:,47)-mean(data_matrix_train(:,47)))./(std(data_matrix_train(:,47),1));
data_matrix_train_standardized(:,48)=(data_matrix_train(:,48)-mean(data_matrix_train(:,48)))./(std(data_matrix_train(:,48),1));
save('data_matrix_train_standardized.mat','data_matrix_train_standardized');
training_mean(1,1)=mean(data_matrix_train(:,1));
training_mean(1,2)=mean(data_matrix_train(:,10));
training_mean(1,3)=mean(data_matrix_train(:,11));
training_mean(1,4)=mean(data_matrix_train(:,46));
training_mean(1,5)=mean(data_matrix_train(:,47));
training_mean(1,6)=mean(data_matrix_train(:,48));
save('training_mean.mat','training_mean');
training_std(1,1)=std(data_matrix_train(:,1),1);
training_std(1,2)=std(data_matrix_train(:,10),1);
training_std(1,3)=std(data_matrix_train(:,11),1);
training_std(1,4)=std(data_matrix_train(:,46),1);
training_std(1,5)=std(data_matrix_train(:,47),1);
training_std(1,6)=std(data_matrix_train(:,48),1);
save('training_std.mat','training_std');

