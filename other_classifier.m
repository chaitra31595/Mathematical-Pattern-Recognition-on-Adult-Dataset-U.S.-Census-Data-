load('data_matrix_train.mat');
load('data_matrix_train_standardized.mat');
load('balanced_data_train.mat');
load('balanced_data_train_standardized.mat')
load('data_matrix_test.mat');
load('data_matrix_test_standardized.mat');
load('data_matrix_test_standardized_smote.mat');

% cv1=cvpartition(length(data_matrix_train(:,end)),'holdout',0.4);
% cv2=cvpartition(length(data_matrix_test(:,end)),'holdout',0.4);
% % cv1 = cvpartition(data_matrix_train(:,end),'KFold',5,'Stratify',true);
% % cv2 = cvpartition(data_matrix_test(:,end),'KFold',5,'Stratify',true);
% X1=data_matrix_train(:,1:end-1);
% Y1=data_matrix_train(:,end);
% X2=data_matrix_test(:,1:end-1);
% Y2=data_matrix_test(:,end);
% % Training set
% Xtrain = X1(training(cv1),:);
% Ytrain = Y1(training(cv1),:);
% % Test set
% Xtest = X2(test(cv2),:);
% Ytest = Y2(test(cv2),:);

% disp('Training Set')
% tabulate(Ytrain)
% disp('Test Set')
% tabulate(Ytest)

% Train the classifier
X=balanced_data_train_standardized;
Y=data_matrix_test_standardized_smote;

disp('Fischer Discriminant Analysis');
Mdl = fitcdiscr(X(:,1:end-1),X(:,end));
Training_Accuracy=sum(Mdl.predict(X(:,1:end-1))==X(:,end))/length(X)
Y_da = Mdl.predict(Y(:,1:end-1));
Testing_Accuracy=sum(Mdl.predict(Y(:,1:end-1))==Y(:,end))/length(Y)
C_da = confusionmat(Y(:,end),Y_da)
Precision=C_da(1,1)/(C_da(1,1)+C_da(1,2))
Recall=C_da(1,1)/(C_da(1,1)+C_da(2,1))
F_score=(2*Recall*Precision)/(Recall+Precision)
C_da1=C_da;
% C_da = bsxfun(@rdivide,C_da,sum(C_da,2)) * 100

disp('KNN Classifier');
Mdl = fitcknn(X(:,1:end-1),X(:,end));
Training_Accuracy=sum(Mdl.predict(X(:,1:end-1))==X(:,end))/length(X)
Y_da = Mdl.predict(Y(:,1:end-1));
Testing_Accuracy=sum(Mdl.predict(Y(:,1:end-1))==Y(:,end))/length(Y)
C_da = confusionmat(Y(:,end),Y_da)
Precision=C_da(1,1)/(C_da(1,1)+C_da(1,2))
Recall=C_da(1,1)/(C_da(1,1)+C_da(2,1))
F_score=(2*Recall*Precision)/(Recall+Precision)
% C_da = bsxfun(@rdivide,C_da,sum(C_da,2)) * 100
C_knn=C_da;

disp('Binary Tree Classifier');
Mdl = fitctree(X(:,1:end-1),X(:,end));
Training_Accuracy=sum(Mdl.predict(X(:,1:end-1))==X(:,end))/length(X)
Y_da = Mdl.predict(Y(:,1:end-1));
Testing_Accuracy=sum(Mdl.predict(Y(:,1:end-1))==Y(:,end))/length(Y)
C_da = confusionmat(Y(:,end),Y_da)
Precision=C_da(1,1)/(C_da(1,1)+C_da(1,2))
Recall=C_da(1,1)/(C_da(1,1)+C_da(2,1))
F_score=(2*Recall*Precision)/(Recall+Precision)
% C_da = bsxfun(@rdivide,C_da,sum(C_da,2)) * 100
C_tb=C_da;

disp('Ensembler');
Mdl = fitcensemble(X(:,1:end-1),X(:,end));
Training_Accuracy=sum(Mdl.predict(X(:,1:end-1))==X(:,end))/length(X)
Y_da = Mdl.predict(Y(:,1:end-1));
Testing_Accuracy=sum(Mdl.predict(Y(:,1:end-1))==Y(:,end))/length(Y)
C_da = confusionmat(Y(:,end),Y_da)
Precision=C_da(1,1)/(C_da(1,1)+C_da(1,2))
Recall=C_da(1,1)/(C_da(1,1)+C_da(2,1))
F_score=(2*Recall*Precision)/(Recall+Precision)
% C_da = bsxfun(@rdivide,C_da,sum(C_da,2)) * 100
C_t=C_da;

Cmat = [ C_da1 C_knn C_t C_tb];
labels = { 'Discriminant Analysis ','k-nearest Neighbors ','Binary Tree ', 'Ensemble '};

% comparisonPlot( Cmat, labels )



% Mdl = fitcknn(data_matrix_train(:,1:end-1),data_matrix_train(:,end));
% Y_da = Mdl.predict(data_matrix_test(:,1:end-1));
% C_da = confusionmat(data_matrix_test(:,end),Y_da);
% C_da = bsxfun(@rdivide,C_da,sum(C_da,2)) * 100
% 
%  Mdl = fitctree(data_matrix_train(:,1:end-1),data_matrix_train(:,end));
% Y_da = Mdl.predict(data_matrix_test(:,1:end-1));
% C_da = confusionmat(data_matrix_test(:,end),Y_da);
% C_da = bsxfun(@rdivide,C_da,sum(C_da,2)) * 100
% 
% Mdl = fitcensemble(data_matrix_train(:,1:end-1),data_matrix_train(:,end));
% Y_da = Mdl.predict(data_matrix_test(:,1:end-1));
% C_da = confusionmat(data_matrix_test(:,end),Y_da);
% C_da = bsxfun(@rdivide,C_da,sum(C_da,2)) * 100
