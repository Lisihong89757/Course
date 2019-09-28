#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:01:17 2019

@author: appleuser

python Xgboost-train.py 
"""

import pandas as pd
import numpy as np
import scipy.stats
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_svmlight_file
import os
import sys
import csv
from collections import defaultdict




#h2o 
#import h2o
# =============================================================================
# train, An_train = zip(*sorted(zip(i_train, An_train)))
# test, An_test = zip(*sorted(zip(i_test, An_test)))
# n = 0
# n_1 = 0
# n_2 = 0
# l = ['A0','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15']
# cname = result.columns
# 
# for each in tp:
#     if n <= 892:
#         if n in train or n in test:
#             each.drop('Antibiotic',axis=1,inplace = True)
#             if each['MIC'].astype(str).str.contains('/').values:
#                 each['MIC'] = each['MIC'].str.split('/').str[0]
#             result = each.iloc[:,2:].astype(np.uint8)
#             result['MIC'] = each['MIC'].astype(np.float32)
#             del each
#             for i in l:
#                 result[i] = np.uint8(0)
#             if test[n_2] - train[n_1] > 0 or n >= 870 :
#                 result[l[An_train[n_1]]] += 1
#                 if not os.path.isfile('staphk10_train.csv'):
#                     result.to_csv('staphk10_train_11.csv', header= cname, index = False)
#                 else: # else it exists so append without writing the header
#                     result.to_csv('staphk10_train.csv', mode='a', header=False, index = False)
#                 n_1 += 1
#                 if n_1 == 732:
#                     break
#             else:
#                 result[l[An_test[n_2]]] += 1
#                 
#                 if not os.path.isfile('staphk10_train.csv'):.
#                     result.to_csv('staphk10_test.csv', header= cname, index = False)
#                 else: # else it exists so append without writing the header
#                     result.to_csv('staphk10_test.csv', mode='a', header=False, index = False)
#                 if n_2 == 81:
#                     n += 1
#                     continue
#                 n_2 += 1 
#                 
#             n += 1
#         else:
#             n += 1
#             continue
#         print(n,n_1,n_2)
#     else:
#         break
#     
#     
# h2o.init(nthreads = -1, max_mem_size = 16)
# data_train_1 = h2o.import_file('staphk10_train_1.csv',col_types=["int" for _ in range(381300)])
# data_train_2 = h2o.import_file('staphk10_train_2.csv',col_types=["int" for _ in range(381300)])
# data_train = data_train_1.cbind(data_train_2)
# data_train_3 = h2o.import_file('staphk10_train_3.csv',col_types=["int" for _ in range(285976)])
# data_train = data_train.cbind(data_train_3)
# data_train_4 = h2o.import_file('staphk10_train_4.csv')
# data_train = data_train.cbind(data_train_4)
# 
# cv_xgb = H2OXGBoostEstimator(
#     ntrees = 100,
#     max_depth = 9,
#     learn_rate = 0.1,
#     max_leaves = 50,
#     stopping_rounds = 10,
#     score_tree_interval = 1,
#     tree_method="auto",
#     grow_policy="depthwise",
#     nfolds=9, 
#     seed=0)
# features = data_train.columns
# features.remove('MIC')
# cv_xgb.train(x = features, y = 'MIC', training_frame = data_train)
# p = cv_xgb.predict(data_test)
# predict = np.array(p.as_data_frame())
# i = data_test[:,'MIC_new']
# i = np.array(i.as_data_frame())
# 
# accuracy_dilute(predict,i)
#                 
# ============================================================================




# ===============================================================
def process(df):
    '''
    this function can change the antibiotic/MIC combination to only one antibiotic/MIC
    'a/b' -> 'a', 'MIC1/MIC2' -> 'MIC1'
    then create two column: 'Antibiotic_new', 'MIC_new' at the end of the dataframe
    '''
    df['Antibiotic_new'] = df['Antibiotic'].str.split('/').str[0]
    df['MIC'] = df['MIC'].astype(str)
    df['MIC_new'] = df['MIC'].str.split('/').str[0]
    df = df.drop(['MIC','Antibiotic'], axis=1)
    return df


def pre_process(df):
    #remove all the low frequecy antibiotics
    freq = df['Antibiotic_new'].value_counts()
    # you can change the threshold here
    threshold = 20
    to_remove = freq[freq < threshold].index
    df['Antibiotic_new'].replace(to_remove, np.nan, inplace=True)
    df = df[pd.notnull(df['Antibiotic_new'])]
    
    df = df.iloc[:,1:]
    (_,tar_col) = df.shape;
    tar_col = tar_col-1
    
    X = df.iloc[:,:tar_col].values;
    Y = df.iloc[:,tar_col].values;
    Y = Y.astype(np.float32);
    
    # label encode
    labelencoder_X = LabelEncoder();
    X[:,tar_col-1] = labelencoder_X.fit_transform(X[:,tar_col-1])
    df.Antibiotic_new = labelencoder_X.fit_transform(df.Antibiotic_new)
    
    Antibiotic_idx = X[:,tar_col-1];
    Antibiotic_idx = list(Antibiotic_idx)
    Antibiotic_idx = np.array(Antibiotic_idx)
    #One-hot encoode
    onehotencoder = OneHotEncoder(categorical_features=[tar_col-1],n_values='auto', 
                                  handle_unknown='ignore',dtype=np.uint16);
    X = onehotencoder.fit_transform(X).toarray()
    X = X.astype(np.int);
    
    df = onehotencoder.fit_transform(df)
    
    return df, Antibiotic_idx, X, Y

# calculate the accuracy based on the ±2 dilution
def accuracy_dilute(prdc,target):
    total = len(prdc);
    n = 0
    for i,j in zip(prdc,target):
        if j/2 <= i and i <= 2*j:
            n += 1  
    return n/total

# calculate the mean accuracy of all the models with 95% confidence interval
# a=[0.5352112676056338, 0.5633802816901409, 0.5549295774647888, 0.5746478873239437, 0.5690140845070423, 0.5464788732394367, 0.5915492957746479, 0.5549295774647888, 0.5211267605633803]
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def xgb_training(df):
    df = process(df)
    df, Antibiotic_idx, X, Y = pre_process(df)

    ##Stratified croess-validation
    acc_list = [];
    idx = list(range(df.shape[0]))
    
    #set up stratified k-fold
    kfold = 9
    skf = StratifiedKFold(n_splits=kfold);

    #seperate the test data from the overall dataset                         
    X_train, X_test, Y_train, Y_test,An_train,An_test= train_test_split(X,Y,Antibiotic_idx,test_size = 0.1, random_state = 1,stratify = Antibiotic_idx);
    
    params = {
        'eta': 0.1,
        'objective':'reg:linear',
        'seed': 99,
        'nthread': 4,
        'max_depth': 10,
        'alpha':0,
        'lambda': 1
        }
    
    n_estimators = 100
    
    #10-fold cross validation
    for i, (train_index, test_index) in enumerate(skf.split(X_train, An_train)):
        print('[Fold %d/%d]' % (i + 1, kfold));
        X_train_v, X_validate = X_train[train_index], X_train[test_index];
        Y_train_v, Y_validate = Y_train[train_index], Y_train[test_index];
        
        d_train = xgb.DMatrix(X_train_v,Y_train_v);
        d_validate = xgb.DMatrix(X_validate,Y_validate);
        d_test = xgb.DMatrix(X_test);
        
        watchlist = [(d_train, 'train'), (d_validate, 'valid')];
        
        model = xgb.train(params,d_train,num_boost_round = n_estimators,evals = watchlist);
        Y_predict = model.predict(d_test, ntree_limit=model.best_ntree_limit);
        
        acc = accuracy_dilute(Y_predict,Y_test);
        print(acc);
        acc_list.append(acc);
    
    print(acc_list)
    acc = mean_confidence_interval(acc_list), X_train, Y_train
    return acc


###############################################################################
#grid search (max_depth, min_child_weight)
###############################################################################
#This is the customized accuracy function which can fit into the model train using argument 'feval'
def accuracy(prdc,target):
    total = len(prdc);
    n = 0
    for i,j in zip(prdc,target.get_label()):
        if j/2 < i and i < 2*j:
            n += 1
    return 'acc', n/total

def xgb_gs(X_train, Y_train):
    
    params = {
        # Parameters that we are going to tune.
        'max_depth':9,
        'min_child_weight': 1,
        'eta':0.1,
        'subsample': 1,
        'colsample_bytree': 1,
        'alpha': 0,
        'lambda': 1,
        'scale_pos_weight':1,
        # Other parameters
        'objective':'reg:linear',
    }
    
    ##############
    # todo
    #You can change the paramemters and and their range here (max_depth, min_child_weight, colsample_bytree, subsample)
    gridsearch_params = [
        (max_depth, min_child_weight,alpha,Lambda)
        for max_depth in [9,15,20,25,30]
        for min_child_weight in range(1,3)
        for alpha in [0.1,0.2,0.3,0.4,0.5]
        for Lambda in range(1,4)]
    
    d_train = xgb.DMatrix(X_train,Y_train);
    num_boost_round = 500
    
    # max accuracy
    max_acc = 0
    best_params = None
    for max_depth, min_child_weight,alpha,Lambda in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}, alpha={}, Lambda={}".format(
                                 max_depth,
                                 min_child_weight,
                                 alpha,
                                 Lambda
                                 ))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        params['alpha'] = alpha
        params['lambda'] = Lambda
        
        # Run CV
        cv_results = xgb.cv(
            params,
            d_train,
            num_boost_round=num_boost_round,
            nfold=10,
            early_stopping_rounds=100,
            #here is the cutomized function
            feval = accuracy
        )
        
        # Update best ACC
        mean_acc = cv_results['test-acc-mean'].max()
        boost_rounds = cv_results['test-acc-mean'].argmax()
        print("\tMAE {} for {} rounds".format(mean_acc, boost_rounds))
        if mean_acc > max_acc:
            max_acc = mean_acc
            best_params = (max_depth,min_child_weight,alpha,Lambda)
            
    #print out the best result we can get
    print("Best params: max_depth {}, min_child_weight {}, alpha {}, Lambda {}, ACC: {}".format(best_params[0], best_params[1],best_params[2],best_params[3],max_acc))
    
    

# this is a function which can reduce the size of the raw data
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props

def main():
    # get the current path
    cwd = os.getcwd()
    
    # todo
    # ========================================================================
    # if you have several files to load, use the code below
    #df = pd.DataFrame()
    
    #for file in os.listdir(cwd):
        #if file.endswith(".tsv"):
            #df_temp = pd.read_csv(cwd+'\\'+file, "\t")
            #df = pd.concat([df, df_temp])
    # ========================================================================
    
    # if you have only one file to load, use the code below
    # remember to change your file name  
    df = pd.read_csv(cwd + "/TRAININGDATAStaphylococcusAureusK3.tsv", "\t");
    
    # get rid of wrong data rows
    df = df[df.MIC != '0.0']
    df = df[df.MIC != 0]
    df = df[df.MIC != 'MIC']
    
    # get rid of wierd features
    names = [col for col in df.columns if 'N' not in col]
    df = df[names]
    
    # this function is at the end of the file, usually you don't need to do this
    # df = reduce_mem_usage(df)
    
    # depend on the data, in the MIC column, it might have '-' in some data point so you can remove it with this code
    # df = df[df['MIC'].map(lambda x: False if '-' in x else True)];
    
    # training
    acc, x_train, y_train = xgb_training(df)
    print(acc)
    
    # grid search
    xgb_gs(x_train, y_train)

if __name__ == "__main__":
    main()
    
    
