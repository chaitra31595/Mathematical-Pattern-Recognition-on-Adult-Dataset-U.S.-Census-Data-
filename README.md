# Mathematical-Pattern-Recognition-on-Adult-Dataset-U.S.-Census-Data-


Features have been taken from U.S. Census data, and include age, education,
occupation category, etc. This is a 2-class problem.
Datasets for training and testing have been down-sampled by factor of 3 (stratified), from the full-dataset versions without missing data on the UCI web site.
For more information on the dataset, and to access the full dataset:
https://archive.ics.uci.edu/ml/datasets/Adult

# Feature Engineering

The features include the following information of a person: age, education, marital status, working class, occupation, relationship, race, sex, capital gain, capital loss, hours per week, native country of an individual and his/her associated annual income.

# Pre-processing
1. Data Parsing
2. Recasting of data representation of features with ordered/unordered categorical to one-hot encoding
3. Missing values in the feature set are compensated/filled using k nearest neighbor approach
4. Compensate the unbalanced data set by adopting SMOTE (Synthetic Minority Oversampling Techniques)
5. Standardizing/ Normalizing the data

# Training and Classification 
- 10,853 samples (unbalanced data) are used for training, out of which 8,240 sample belongs to “<=50K” label and 2,613 sample belongs to “>50K” label.
- To oversample the minority class label “>50K” samples, SMOTE is adopted by generating 2 samples for every minority class training sample.
- Generated Balanced dataset contains 16,079 training samples, out of which 8,240 training sample belongs to “<=50K” label and 7,839 sample belongs to “>50K” label.
- Test data contains 5,427 samples, out of which 4,145 sample belongs to “<=50K” label and 1,282 sample belongs to “>50K” label.
- Combinations of unbalanced (or balanced) data sets with (or without) standardization are used for training the model.
- Testing data with (or without) standardization using the corresponding training sample mean and standard deviation are used while testing data.
- 5 classifiers: Support Vector Machine, Fischer discriminant analysis, KNN (K nearest neighbor), binary tree classifier, Ensembler are trained and tested under various conditions.
- Corresponding training and testing accuracy, confusion matrix, F score are tabulated and interpreted.





