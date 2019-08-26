load('adult_test.mat')
load('training_mean');
load('training_std');
load('balanced_data_training_mean');
load('balanced_data_training_std');
adult_test=adult2;
[data_size features]=size(adult_test);
data_matrix_test(:,1)=adult_test{:,1};   % Age
data_matrix_test(:,10)=adult_test{:,3};  % fnlwt
data_matrix_test(:,11)=adult_test{:,5};  % education-num
data_matrix_test(:,46)=adult_test{:,11}; % capital-gain
data_matrix_test(:,47)=adult_test{:,12}; % capital-loss
data_matrix_test(:,48)=adult_test{:,13}; % hours-per-week

% nation=["Cambodia" "Canada" "China" "Columbia" "Cuba" "Dominican-Republic" "Ecuador" "El-Salvador" "England" "France" "Germany" "Greece" "Guatemala" "Haiti" "Honduras" "Hong" "Hungary" "India" "Iran" "Ireland" "Italy" "Jamaica" "Japan" "Laos" "Mexico" "Nicaragua" "Outlying-US(Guam-USVI-etc)" "Peru" "Philippines" "Poland" "Portugal" "Puerto-Rico" "Scotland" "South" "Taiwan" "Thailand" "Trinadad&Tobago" "United-States" "Vietnam" "Yugoslavia"];
% workclass=["Private" "Self-emp-not-inc" "Self-emp-inc" "Federal-gov" "Local-gov" "State-gov" "Without-pay" "Never-worked"];
% marital_status=["Married-civ-spouse" "Divorced" "Never-married" "Separated" "Widowed" "Married-spouse-absent" "Married-AF-spouse"];
% occupation=["Tech-support" "Craft-repair" "Other-service" "Sales" "Exec-managerial" "Prof-specialty" "Handlers-cleaners" "Machine-op-inspct" "Adm-clerical" "Farming-fishing" "Transport-moving" "Priv-house-serv" "Protective-serv" "Armed-Forces"];
% relationship=["Wife" "Own-child" "Husband" "Not-in-family" "Other-relative" "Unmarried"];
% race=["White" "Asian-Pac-Islander" "Amer-Indian-Eskimo" "Other" "Black"]; 
% sex=[ "Female" "Male"];
nation=[" Cambodia" " Canada" " China" " Columbia" " Cuba" " Dominican-Republic" " Ecuador" " El-Salvador" " England" " France" " Germany" " Greece" " Guatemala" " Haiti" " Honduras" " Hong" " Hungary" " India" " Iran" " Ireland" " Italy" " Jamaica" " Japan" " Laos" " Mexico" " Nicaragua" " Outlying-US(Guam-USVI-etc)" " Peru" " Philippines" " Poland" " Portugal" " Puerto-Rico" " Scotland" " South" " Taiwan" " Thailand" " Trinadad&Tobago" " United-States" " Vietnam" " Yugoslavia"];
workclass=[" Private" " Self-emp-not-inc" " Self-emp-inc" " Federal-gov" " Local-gov" " State-gov" " Without-pay" " Never-worked"];
marital_status=[" Married-civ-spouse" " Divorced" " Never-married" " Separated" " Widowed" " Married-spouse-absent" " Married-AF-spouse"];
occupation=[" Tech-support" " Craft-repair" " Other-service" " Sales" " Exec-managerial" " Prof-specialty" " Handlers-cleaners" " Machine-op-inspct" " Adm-clerical" " Farming-fishing" " Transport-moving" " Priv-house-serv" " Protective-serv" " Armed-Forces"];
relationship=[" Wife" " Own-child" " Husband" " Not-in-family" " Other-relative" " Unmarried"];
race=[" White" " Asian-Pac-Islander" " Amer-Indian-Eskimo" " Other" " Black"]; 
sex=[ " Female" " Male"];
k=1;l=1;m=1;n=1;o=1;p=1;q=1;
for i=1:data_size
    index1 = find(contains(workclass,adult_test{i,2}));
    if(~isempty(index1))
    data_matrix_test(i,1+index1)=1;
    else
       missing_index(1,k)=i;
       k=k+1;
    end

    index2 = find(contains(marital_status,adult_test{i,6}));
    if(~isempty(index2))
    data_matrix_test(i,11+index2)=1; 
    else
       missing_index(2,l)=i;
       l=l+1;
    end

    index3 = find(contains(occupation,adult_test{i,7}));
    if(~isempty(index3))
    data_matrix_test(i,18+index3)=1;
    else
       missing_index(3,m)=i;
       m=m+1;
    end
        
    index4 = find(contains(relationship,adult_test{i,8}));
    if(~isempty(index4))
    data_matrix_test(i,32+index4)=1;
    else
       missing_index(4,n)=i;
       n=n+1;
    end    

     index5 = find(contains(race,adult_test{i,9}));
    if(~isempty(index5))
    data_matrix_test(i,38+index5)=1;
    else
       missing_index(5,o)=i;
       o=o+1;
    end   

     index6 = find(contains(sex,adult_test{i,10}));
    if(~isempty(index6))
    data_matrix_test(i,43+index6)=1;
    else
       missing_index(6,p)=i;
       p=p+1;
    end
    
    index7 = find(contains(nation,adult_test{i,14}));
    if(~isempty(index7))
    data_matrix_test(i,48+index7)=1;
    else
       missing_index(7,q)=i;
       q=q+1;
    end
        
    if(strcmp(adult_test{i,15}," <=50K."))
       data_matrix_test(i,89)=-1;
    else
       data_matrix_test(i,89)=1;
    end
end


if(k>1)  %identifying missing values for workclass
for r=1:k-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(1,:)==i)) && (i~=missing_index(1,r)))
        dist1(r,j)=sqrt(sum((data_matrix_test(missing_index(1,r),1)-data_matrix_test(i,1)).^2+(data_matrix_test(missing_index(1,r),10:end)-data_matrix_test(i,10:end)).^2));
    else
        dist1(r,j)=Inf;
    end
    j=j+1;
end
[min_value,ind]=min(dist1(r,:));
in = find(contains(workclass,adult_test{ind,2}));
data_matrix_test(missing_index(1,r),1+in)=1;
end
end

if(l>1)  %identifying missing values for marital_status
for r=1:l-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(2,:)==i)) && (i~=missing_index(2,r)))
        dist2(r,j)=sqrt(sum((data_matrix_test(missing_index(2,r),1:11)-data_matrix_test(i,1:11)).^2)+sum((data_matrix_test(missing_index(2,r),19:end)-data_matrix_test(i,19:end)).^2));
    else
        dist2(r,j)=Inf;
    end   
    j=j+1;     
end
[min_value,ind]=min(dist2(r,:));
in = find(contains(marital_status,adult_test{ind,6}));
data_matrix_test(missing_index(2,r),11+in)=1;
end
end
if(m>1)  %identifying missing values for occupation
for r=1:m-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(3,:)==i)) && (i~=missing_index(3,r)))
        dist3(r,j)=sqrt(sum((data_matrix_test(missing_index(3,r),1:18)-data_matrix_test(i,1:18)).^2)+sum((data_matrix_test(missing_index(3,r),33:end)-data_matrix_test(i,33:end)).^2));
    else
        dist3(r,j)=Inf;
    end   
    j=j+1;     
end
[min_value,ind]=min(dist3(r,:));
in = find(contains(occupation,adult_test{ind,7}));
data_matrix_test(missing_index(3,r),18+in)=1;
end
end
if(n>1)  %identifying missing values for relationship
for r=1:n-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(4,:)==i)) && (i~=missing_index(4,r)))
        dist4(r,j)=sqrt(sum((data_matrix_test(missing_index(4,r),1:32)-data_matrix_test(i,1:32)).^2)+sum((data_matrix_test(missing_index(4,r),39:end)-data_matrix_test(i,39:end)).^2));
    else
        dist4(r,j)=Inf;
    end   
    j=j+1;     
end
[min_value,ind]=min(dist4(r,:));
in = find(contains(relationship,adult_test{ind,8}));
data_matrix_test(missing_index(4,r),32+in)=1;
end
end
if(o>1)  %identifying missing values for race
for r=1:o-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(5,:)==i)) && (i~=missing_index(5,r)))
        dist5(r,j)=sqrt(sum((data_matrix_test(missing_index(5,r),1:38)-data_matrix_test(i,1:38)).^2)+sum((data_matrix_test(missing_index(5,r),44:end)-data_matrix_test(i,44:end)).^2));
    else
        dist5(r,j)=Inf;
    end   
    j=j+1;     
end
[min_value,ind]=min(dist5(r,:));
in = find(contains(race,adult_test{ind,9}));
data_matrix_test(missing_index(5,r),38+in)=1;
end
end
if(p>1)  %identifying missing values for sex
for r=1:p-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(6,:)==i)) && (i~=missing_index(6,r)))
        dist6(r,j)=sqrt(sum((data_matrix_test(missing_index(6,r),1:43)-data_matrix_test(i,1:43)).^2)+sum((data_matrix_test(missing_index(6,r),46:end)-data_matrix_test(i,46:end)).^2));
    else
        dist6(r,j)=Inf;
    end   
    j=j+1;     
end
[min_value,ind]=min(dist6(r,:));
in = find(contains(sex,adult_test{ind,10}));
data_matrix_test(missing_index(6,r),43+in)=1;
end
end
if(q>1)  %identifying missing values for nation
for r=1:q-1
    j=1;
for i=1:data_size
    if(isempty(find(missing_index(7,:)==i)) && (i~=missing_index(7,r)))
        dist7(r,j)=sqrt(sum((data_matrix_test(missing_index(7,r),1:48)-data_matrix_test(i,1:48)).^2)+sum((data_matrix_test(missing_index(7,r),end)-data_matrix_test(i,end)).^2));
    else
        dist7(r,j)=Inf;
    end   
    j=j+1;
end
[min_value,ind]=min(dist7(r,:));
in = find(contains(nation,adult_test{ind,14}));
data_matrix_test(missing_index(7,r),48+in)=1;
end
end

data_matrix_test=data_matrix_test;
save('data_matrix_test.mat','data_matrix_test');

data_matrix_test_standardized=data_matrix_test;
data_matrix_test_standardized(:,1)=(data_matrix_test(:,1)-training_mean(1,1))./(training_std(1,1));
data_matrix_test_standardized(:,10)=(data_matrix_test(:,10)-training_mean(1,2))./(training_std(1,2));
data_matrix_test_standardized(:,11)=(data_matrix_test(:,11)-training_mean(1,3))./(training_std(1,3));
data_matrix_test_standardized(:,46)=(data_matrix_test(:,46)-training_mean(1,4))./(training_std(1,4));
data_matrix_test_standardized(:,47)=(data_matrix_test(:,47)-training_mean(1,5))./(training_std(1,5));
data_matrix_test_standardized(:,48)=(data_matrix_test(:,48)-training_mean(1,6))./(training_std(1,6));
save('data_matrix_test_standardized.mat','data_matrix_test_standardized');

data_matrix_test_standardized_smote=data_matrix_test;
data_matrix_test_standardized_smote(:,1)=(data_matrix_test(:,1)-balanced_data_training_mean(1,1))./(balanced_data_training_std(1,1));
data_matrix_test_standardized_smote(:,10)=(data_matrix_test(:,10)-balanced_data_training_mean(1,2))./(balanced_data_training_std(1,2));
data_matrix_test_standardized_smote(:,11)=(data_matrix_test(:,11)-balanced_data_training_mean(1,3))./(balanced_data_training_std(1,3));
data_matrix_test_standardized_smote(:,46)=(data_matrix_test(:,46)-balanced_data_training_mean(1,4))./(balanced_data_training_std(1,4));
data_matrix_test_standardized_smote(:,47)=(data_matrix_test(:,47)-balanced_data_training_mean(1,5))./(balanced_data_training_std(1,5));
data_matrix_test_standardized_smote(:,48)=(data_matrix_test(:,48)-balanced_data_training_mean(1,6))./(balanced_data_training_std(1,6));
save('data_matrix_test_standardized_smote.mat','data_matrix_test_standardized_smote');
