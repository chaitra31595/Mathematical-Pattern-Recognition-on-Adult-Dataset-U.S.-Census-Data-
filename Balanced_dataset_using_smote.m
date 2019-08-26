load('data_matrix_train.mat')
data_matrix=data_matrix_train;
count_for_greater_than_50K=sum(data_matrix(:,end)==1);
count_for_lesser_than_or_equal_50K=sum(data_matrix(:,end)==-1);

if(count_for_greater_than_50K>count_for_lesser_than_or_equal_50K)
    minority_class_label=-1;
else
    minority_class_label=1; 
end

s=1;
for i=1:length(data_matrix)
   if(data_matrix(i,end)==minority_class_label)
       minority_class_index(1,s)=i;
       for j=1:length(data_matrix)
         if(j~=i)  
          nearest_neigbor(s,j)=sqrt(sum((data_matrix(i,1:end-1)-data_matrix(j,1:end-1)).^2));
         else
          nearest_neigbor(s,j)=Inf;
         end
       end
      s=s+1;
   end
end

% [val,index]=min(nearest_neigbor,[],2);
[val,index]=mink(nearest_neigbor',2);
h=1;
for i=1:length(minority_class_index) 
    oversampled_data(h,:)=data_matrix(minority_class_index(1,i),1:end-1)+rand*(data_matrix(minority_class_index(1,i),1:end-1)-data_matrix(index(1,i),1:end-1));
    oversampled_data(h+1,:)=data_matrix(minority_class_index(1,i),1:end-1)+rand*(data_matrix(minority_class_index(1,i),1:end-1)-data_matrix(index(2,i),1:end-1));
    h=h+2;
end

oversampled_data(:,89)=minority_class_label;

balanced_data_train=data_matrix;
balanced_data_train(length(data_matrix)+1:length(data_matrix)+length(oversampled_data),:)=oversampled_data;
save('balanced_data_train.mat','balanced_data_train');


balanced_data_train_standardized=balanced_data_train;
balanced_data_train_standardized(:,1)=(balanced_data_train(:,1)-mean(balanced_data_train(:,1)))./(std(balanced_data_train(:,1),1));
balanced_data_train_standardized(:,10)=(balanced_data_train(:,10)-mean(balanced_data_train(:,10)))./(std(balanced_data_train(:,10),1));
balanced_data_train_standardized(:,11)=(balanced_data_train(:,11)-mean(balanced_data_train(:,11)))./(std(balanced_data_train(:,11),1));
balanced_data_train_standardized(:,46)=(balanced_data_train(:,46)-mean(balanced_data_train(:,46)))./(std(balanced_data_train(:,46),1));
balanced_data_train_standardized(:,47)=(balanced_data_train(:,47)-mean(balanced_data_train(:,47)))./(std(balanced_data_train(:,47),1));
balanced_data_train_standardized(:,48)=(balanced_data_train(:,48)-mean(balanced_data_train(:,48)))./(std(balanced_data_train(:,48),1));
save('balanced_data_train_standardized.mat','balanced_data_train_standardized');


balanced_data_training_mean(1,1)=mean(balanced_data_train(:,1));
balanced_data_training_mean(1,2)=mean(balanced_data_train(:,10));
balanced_data_training_mean(1,3)=mean(balanced_data_train(:,11));
balanced_data_training_mean(1,4)=mean(balanced_data_train(:,46));
balanced_data_training_mean(1,5)=mean(balanced_data_train(:,47));
balanced_data_training_mean(1,6)=mean(balanced_data_train(:,48));
save('balanced_data_training_mean.mat','balanced_data_training_mean');
balanced_data_training_std(1,1)=std(balanced_data_train(:,1),1);
balanced_data_training_std(1,2)=std(balanced_data_train(:,10),1);
balanced_data_training_std(1,3)=std(balanced_data_train(:,11),1);
balanced_data_training_std(1,4)=std(balanced_data_train(:,46),1);
balanced_data_training_std(1,5)=std(balanced_data_train(:,47),1);
balanced_data_training_std(1,6)=std(balanced_data_train(:,48),1);
save('balanced_data_training_std.mat','balanced_data_training_std');



