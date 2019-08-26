load('data_matrix_train.mat');
load('data_matrix_train_standardized.mat');
load('balanced_data_train.mat');
load('balanced_data_train_standardized.mat')
load('data_matrix_test.mat');
load('data_matrix_test_standardized.mat');
load('data_matrix_test_standardized_smote.mat');

model1 = svmtrain(data_matrix_train(:,end), data_matrix_train(:,1:end-1),'-t 0 -c 1 ');
[pred_label1,accuracy1, dec_values1] = svmpredict(data_matrix_train(:,end), data_matrix_train(:,1:end-1), model1);
disp('Training unbalanced data without standardization - Accuracy:');
disp(accuracy1(1,1));

[pred_label1_test,accuracy1_test, dec_values1_test] = svmpredict(data_matrix_test(:,end), data_matrix_test(:,1:end-1), model1);
disp('Training unbalanced data without standardization......Testing without standardization');
disp(accuracy1_test(1,1));
[sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
F_score= (2*Recall*Precision)/(Recall+Precision)

model2 = svmtrain(data_matrix_train_standardized(:,end), data_matrix_train_standardized(:,1:end-1),'-t 0 -c 1 ');
[pred_label1,accuracy2, dec_values1] = svmpredict(data_matrix_train_standardized(:,end), data_matrix_train_standardized(:,1:end-1), model2);
disp('Training unbalanced data with standardization - Accuracy:');
disp(accuracy2(1,1));
[pred_label1_test,accuracy2_test, dec_values1_test] = svmpredict(data_matrix_test(:,end), data_matrix_test(:,1:end-1), model2);
disp('Training unbalanced data with standardization......Testing without standardization');
disp(accuracy2_test(1,1));
[sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
F_score= (2*Recall*Precision)/(Recall+Precision)
[pred_label1_test,accuracy2_test, dec_values1_test] = svmpredict(data_matrix_test_standardized(:,end), data_matrix_test_standardized(:,1:end-1), model2);
disp('Training unbalanced data with standardization......Testing with standardization');
disp(accuracy2_test(1,1));
[sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
F_score= (2*Recall*Precision)/(Recall+Precision)


model3 = svmtrain(balanced_data_train(:,end), balanced_data_train(:,1:end-1),'-t 0 -c 1 ');
[pred_label1,accuracy3, dec_values1] = svmpredict(balanced_data_train(:,end), balanced_data_train(:,1:end-1), model3);
disp('Training balanced data without standardization - Accuracy:');
disp(accuracy3(1,1));
[sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
F_score= (2*Recall*Precision)/(Recall+Precision)
[pred_label1_test,accuracy3_test, dec_values1_test] = svmpredict(data_matrix_test(:,end), data_matrix_test(:,1:end-1), model3);
disp('Training balanced data without standardization......Testing without standardization');
disp(accuracy3_test(1,1));
[sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
F_score= (2*Recall*Precision)/(Recall+Precision)

model4 = svmtrain(balanced_data_train_standardized(:,end), balanced_data_train_standardized(:,1:end-1),'-t 0 -c 1 ');
[pred_label1,accuracy4, dec_values1] = svmpredict(balanced_data_train_standardized(:,end), balanced_data_train_standardized(:,1:end-1), model4);
disp('Training balanced data with standardization - Accuracy:');
disp(accuracy4(1,1));
[pred_label1_test,accuracy4_test, dec_values1_test] = svmpredict(data_matrix_test(:,end), data_matrix_test(:,1:end-1), model4);
disp('Training balanced data without standardization......Testing without standardization');
disp(accuracy4_test(1,1));
[sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
F_score= (2*Recall*Precision)/(Recall+Precision)
[pred_label1_test,accuracy4_test, dec_values1_test] = svmpredict(data_matrix_test_standardized_smote(:,end), data_matrix_test_standardized_smote(:,1:end-1), model4);
disp('Training balanced data without standardization......Testing with standardization');
disp(accuracy4_test(1,1));
[sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
F_score= (2*Recall*Precision)/(Recall+Precision)


% disp('CROSS VALIDATION');
% model1 = svmtrain(data_matrix_train(:,end), data_matrix_train(:,1:end-1),'-t 0 -c 1 -v 5');
% [pred_label1,accuracy1, dec_values1] = svmpredict(data_matrix_train(:,end), data_matrix_train(:,1:end-1), model1);
% disp('Training unbalanced data without standardization - Accuracy:');
% disp(accuracy1(1,1));
% [pred_label1_test,accuracy1_test, dec_values1_test] = svmpredict(data_matrix_test(:,end), data_matrix_test(:,1:end-1), model1);
% disp('Training unbalanced data without standardization......Testing without standardization');
% disp(accuracy1_test(1,1));
% [sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
% Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
% Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
% F_score= (2*Recall*Precision)/(Recall+Precision)
% 
% 
% model2 = svmtrain(data_matrix_train_standardized(:,end), data_matrix_train_standardized(:,1:end-1),'-t 0 -c 1 -v 5');
% [pred_label1,accuracy2, dec_values1] = svmpredict(data_matrix_train_standardized(:,end), data_matrix_train_standardized(:,1:end-1), model2);
% disp('Training unbalanced data with standardization - Accuracy:');
% disp(accuracy2(1,1));
% [pred_label1_test,accuracy2_test, dec_values1_test] = svmpredict(data_matrix_test(:,end), data_matrix_test(:,1:end-1), model2);
% disp('Training unbalanced data with standardization......Testing without standardization');
% disp(accuracy2_test(1,1));
% [sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
% Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
% Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
% F_score= (2*Recall*Precision)/(Recall+Precision)
% [pred_label1_test,accuracy2_test, dec_values1_test] = svmpredict(data_matrix_test_standardized(:,end), data_matrix_test_standardized(:,1:end-1), model2);
% disp('Training unbalanced data with standardization......Testing with standardization');
% disp(accuracy2_test(1,1));
% [sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
% Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
% Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
% F_score= (2*Recall*Precision)/(Recall+Precision)
% 
% 
% model3 = svmtrain(balanced_data_train(:,end), balanced_data_train(:,1:end-1),'-t 0 -c 1 -v 5');
% [pred_label1,accuracy3, dec_values1] = svmpredict(balanced_data_train(:,end), balanced_data_train(:,1:end-1), model3);
% disp('Training balanced data without standardization - Accuracy:');
% disp(accuracy3(1,1));
% [pred_label1_test,accuracy3_test, dec_values1_test] = svmpredict(data_matrix_test(:,end), data_matrix_test(:,1:end-1), model3);
% disp('Training balanced data without standardization......Testing without standardization');
% disp(accuracy3_test(1,1));
% [sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
% Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
% Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
% F_score= (2*Recall*Precision)/(Recall+Precision)
% 
% 
% model4 = svmtrain(balanced_data_train_standardized(:,end), balanced_data_train_standardized(:,1:end-1),'-t 0 -c 1 -v 5');
% [pred_label1,accuracy4, dec_values1] = svmpredict(balanced_data_train_standardized(:,end), balanced_data_train_standardized(:,1:end-1), model4);
% disp('Training balanced data with standardization - Accuracy:');
% disp(accuracy4(1,1));
% [pred_label1_test,accuracy4_test, dec_values1_test] = svmpredict(data_matrix_test(:,end), data_matrix_test(:,1:end-1), model4);
% disp('Training balanced data without standardization......Testing without standardization');
% disp(accuracy4_test(1,1));
% [sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
% Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
% Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
% F_score= (2*Recall*Precision)/(Recall+Precision)
% [pred_label1_test,accuracy4_test, dec_values1_test] = svmpredict(data_matrix_test_standardized_smote(:,end), data_matrix_test_standardized_smote(:,1:end-1), model4);
% disp('Training balanced data without standardization......Testing with standardization');
% disp(accuracy4_test(1,1));
% [sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1) sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1) ;  sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1)  sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==-1)]
% Precision=sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==-1),1)==1))
% Recall= sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)/(sum(pred_label1_test((data_matrix_test(:,end)==1),1)==1)+sum(pred_label1_test((data_matrix_test(:,end)==1),1)==-1))
% F_score= (2*Recall*Precision)/(Recall+Precision)
% 
