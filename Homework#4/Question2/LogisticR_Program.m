% Main program to train and test
clc;
clear;

% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations.

% load MatLab data
file_path='C:/Users/raulq/raulS/MS_cse/Semesters/spring2022/cse847_ML/hw/ml_hw4/alzheimers/ad_data.mat';
Struct_file = load(file_path);

% Assign train and test data & labels
train_data=Struct_file.X_train;   % X_train 172x116
label_train_data=Struct_file.y_train;  % y_train 172x1
test_data=Struct_file.X_test;  % X_test 74x116
label_test_data=Struct_file.y_test;  % y_test 74x1

data=train_data;
labels=label_train_data;

% values for par
par =[0.000000000000001, 0.02, 0.03, 0.04,0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
sz=size(par,2);
% initialize lists
AUC_list=[];
number_feature_select=[];
for i=1:sz
    par_i=par(i);
    % function LogisticR
    [w, c, funVal, ValueL]=LogisticR_l1_train(data, labels, par_i, opts);

    % calculate the number of features selected ((number of non-zero entries in w)
    sz_w=size(w,1);
    count=0;
    for j=1:sz_w
        if w(j,1)==0
            continue
        else
            count=count+1;
        end
    end
    number_feature_select(i)=count;

    % Prediction on test_data
    s=test_data*w;
    scores=sign(s);

    % perfcurve Receiver operating characteristic (ROC) curve or other performance curve for classifier output
    posclass=1;
    [X,Y,T,AUC] = perfcurve(label_test_data,scores,posclass);
    AUC_list(i)=AUC;
end

% Results to use for plots in excel
AUC_list_T=transpose(AUC_list);
number_feature_select_T=transpose(number_feature_select);
par_T=transpose(par);









%function [w, c] = logistic l1 train(data, labels, par)