from __future__ import print_function, division

from collections import Counter
import pandas as pd
import random,time,csv
import numpy as np
import copy
import math,copy,os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn import preprocessing
from sklearn.semi_supervised import SelfTrainingClassifier


import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.abspath('..'))

from Measure import measure_final_score,calculate_recall,calculate_far,calculate_precision,calculate_accuracy
from Generate_Samples import generate_samples

def self_training(dataset,protected_attribute,percentage):

    if dataset == "adult":
        
        ## Load dataset
        dataset_orig = pd.read_csv('../data/adult.data.csv')

        ## Drop NULL values
        dataset_orig = dataset_orig.dropna()

        dataset_orig = dataset_orig.drop(['workclass','fnlwgt','marital-status','relationship','native-country'],axis=1)


        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
        dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)

        for col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('category')
            dataset_orig[col] = dataset_orig[col].cat.codes
        

        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

        dataset_orig = dataset_orig.drop_duplicates() ## Remove duplicates

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        # divide the data based on protected_attribute
        dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

        # Check Default model scores on test data


        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        
        print("---------------RESULTS BEFORE Fair-SSL------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Train model for privileged group

        dataset_orig_male[protected_attribute] = 0
        X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
        clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_male.fit(X_train_male, y_train_male)


        # Train model for unprivileged group

        X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
        clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_female.fit(X_train_female, y_train_female)


        # select fair rows

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        for index,row in dataset_orig_train.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_male = clf_male.predict(row_)
            y_female = clf_female.predict(row_)
            if y_male[0] != y_female[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig_train = dataset_orig_train.drop(index)


        # prepare labeled and unlabeled data

        zero_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        zero_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        one_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        one_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        initial_size = len(dataset_orig_train)*percentage//400

        df1 = zero_zero_df[:initial_size].append(zero_one_df[:initial_size])
        df2 = df1.append(one_zero_df[:initial_size])
        df3 = df2.append(one_one_df[:initial_size])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[initial_size:].append(zero_one_df[initial_size:])
        df2 = df1.append(one_zero_df[initial_size:])
        df3 = df2.append(one_one_df[initial_size:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        # Train self-training model

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        self_training_model = SelfTrainingClassifier(clf)
        self_training_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = self_training_model.predict_proba(X_unl)
        y_pred = self_training_model.predict(X_unl)


        to_keep = []

        for i in range(len(y_pred_proba)):
            if max(y_pred_proba[i]) >= 0.6:
                to_keep.append(i)

        X_unl_certain = X_unl.iloc[to_keep,:]
        y_unl_certain = y_pred[to_keep]

        X_train, y_train = labeled_df.loc[:, labeled_df.columns != 'Probability'], labeled_df['Probability']

        X_train = X_train.append(X_unl_certain)
        y_train = np.concatenate([y_train,y_unl_certain])


        # check scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)

        print("---------------RESULTS AFTER Self-training ------------------")


        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Balance data for further improvement

        X_train['Probability'] = y_train
        dataset_orig_train = X_train

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        zero_zero_to_be_incresed = maximum - zero_zero ## where both are 0
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        df_zero_zero['race'] = df_zero_zero['race'].astype(str)
        df_zero_zero['sex'] = df_zero_zero['sex'].astype(str)


        df_one_zero['race'] = df_one_zero['race'].astype(str)
        df_one_zero['sex'] = df_one_zero['sex'].astype(str)

        df_one_one['race'] = df_one_one['race'].astype(str)
        df_one_one['sex'] = df_one_one['sex'].astype(str)


        df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'Adult')
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'Adult')
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'Adult')


        df = df_zero_zero.append(df_one_zero)
        df = df.append(df_one_one)

        df['race'] = df['race'].astype(float)
        df['sex'] = df['sex'].astype(float)


        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df = df.append(df_zero_one)


        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train) # LSR

        print("---------------RESULTS AFTER balancing ------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("=============================================================================")

    if dataset == "compas":

        dataset_orig = pd.read_csv('../data/compas-scores-two-years.csv')


        ## Drop categorical features
        ## Removed two duplicate coumns - 'decile_score','priors_count'
        dataset_orig = dataset_orig.drop(['id','name','first','last','compas_screening_date','dob','age','juv_fel_count','decile_score',
                                          'juv_misd_count','juv_other_count','days_b_screening_arrest','c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date','c_days_from_compas','c_charge_desc',
                                          'is_recid','r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc',
                                          'r_jail_in','r_jail_out','violent_recid','is_violent_recid','vr_case_number','vr_charge_degree',
                                          'vr_offense_date','vr_charge_desc','type_of_assessment','decile_score','score_text','screening_date','v_type_of_assessment',
                                          'v_decile_score','v_score_text','v_screening_date','in_custody','out_custody','start','end','event'],axis=1)

        ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 1, 0)
        dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
        dataset_orig['priors_count'] = np.where((dataset_orig['priors_count'] >= 1 ) & (dataset_orig['priors_count'] <= 3), 3, dataset_orig['priors_count'])
        dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45',45,dataset_orig['age_cat'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
        dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)

        ## Rename class column
        dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)

        ## Here did not rec means 0 is the favorable lable
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)


        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        # divide the data based on protected_attribute
        dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

        # Check Default model scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        
        print("---------------RESULTS BEFORE Fair-SSL------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))

        dataset_orig_male[protected_attribute] = 0
        X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
        clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_male.fit(X_train_male, y_train_male)


        X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
        clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_female.fit(X_train_female, y_train_female)

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        for index,row in dataset_orig_train.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_male = clf_male.predict(row_)
            y_female = clf_female.predict(row_)
            if y_male[0] != y_female[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig_train = dataset_orig_train.drop(index)

        zero_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        zero_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        one_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        one_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        initial_size = len(dataset_orig_train)*percentage//400

        df1 = zero_zero_df[:initial_size].append(zero_one_df[:initial_size])
        df2 = df1.append(one_zero_df[:initial_size])
        df3 = df2.append(one_one_df[:initial_size])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[initial_size:].append(zero_one_df[initial_size:])
        df2 = df1.append(one_zero_df[initial_size:])
        df3 = df2.append(one_one_df[initial_size:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)


        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        self_training_model = SelfTrainingClassifier(clf)
        self_training_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = self_training_model.predict_proba(X_unl)
        y_pred = self_training_model.predict(X_unl)


        to_keep = []

        for i in range(len(y_pred_proba)):
            if max(y_pred_proba[i]) >= 0.6:
                to_keep.append(i)

        X_unl_certain = X_unl.iloc[to_keep,:]
        y_unl_certain = y_pred[to_keep]


        X_train, y_train = labeled_df.loc[:, labeled_df.columns != 'Probability'], labeled_df['Probability']

        X_train = X_train.append(X_unl_certain)
        y_train = np.concatenate([y_train,y_unl_certain])


        # check scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)

        print("---------------RESULTS AFTER Self-training ------------------")


        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        X_train['Probability'] = y_train
        dataset_orig_train = X_train

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        one_zero_to_be_incresed = maximum - one_zero ## where class is 0 attribute is 0
        zero_one_to_be_incresed = maximum - zero_one 
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1


        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]


        df_zero_one['race'] = df_zero_one['race'].astype(str)
        df_zero_one['sex'] = df_zero_one['sex'].astype(str)


        df_one_zero['race'] = df_one_zero['race'].astype(str)
        df_one_zero['sex'] = df_one_zero['sex'].astype(str)

        df_one_one['race'] = df_one_one['race'].astype(str)
        df_one_one['sex'] = df_one_one['sex'].astype(str)

        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'Compas')
        df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'Compas')
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'Compas')


        df = df_zero_one.append(df_one_zero)
        df = df.append(df_one_one)

        df['race'] = df['race'].astype(float)
        df['sex'] = df['sex'].astype(float)

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df = df.append(df_zero_zero)

        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train) # LSR

        print("---------------RESULTS AFTER balancing ------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("=============================================================================")


    if dataset == "MEPS15":
        
        ## Load dataset
        dataset_orig = pd.read_csv('../data/MEPS/h181.csv')

        # ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        dataset_orig = dataset_orig.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                                      'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                                      'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                                      'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                                      'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})


        dataset_orig = dataset_orig[dataset_orig['PANEL'] == 20]
        dataset_orig = dataset_orig[dataset_orig['REGION'] >= 0] # remove values -1
        dataset_orig = dataset_orig[dataset_orig['AGE'] >= 0] # remove values -1
        dataset_orig = dataset_orig[dataset_orig['MARRY'] >= 0] # remove values -1, -7, -8, -9
        dataset_orig = dataset_orig[dataset_orig['ASTHDX'] >= 0] # remove values -1, -7, -8, -9
        dataset_orig = dataset_orig[(dataset_orig[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                     'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                     'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                     'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                     'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]

        # ## Change symbolics to numerics
        dataset_orig['RACEV2X'] = np.where((dataset_orig['HISPANX'] == 2 ) & (dataset_orig['RACEV2X'] == 1), 1, dataset_orig['RACEV2X'])
        dataset_orig['RACEV2X'] = np.where(dataset_orig['RACEV2X'] != 1 , 0, dataset_orig['RACEV2X'])
        dataset_orig = dataset_orig.rename(columns={"RACEV2X" : "RACE"})
        # dataset_orig['UTILIZATION'] = np.where(dataset_orig['UTILIZATION'] >= 10, 1, 0)



        def utilization(row):
                return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

        dataset_orig['TOTEXP15'] = dataset_orig.apply(lambda row: utilization(row), axis=1)
        lessE = dataset_orig['TOTEXP15'] < 10.0
        dataset_orig.loc[lessE,'TOTEXP15'] = 0.0
        moreE = dataset_orig['TOTEXP15'] >= 10.0
        dataset_orig.loc[moreE,'TOTEXP15'] = 1.0

        dataset_orig = dataset_orig.rename(columns = {'TOTEXP15' : 'UTILIZATION'})

        dataset_orig = dataset_orig[['REGION','AGE','SEX','RACE','MARRY',
                                         'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                         'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                         'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                         'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                         'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION', 'PERWT15F']]

        dataset_orig = dataset_orig.rename(columns={"UTILIZATION": "Probability","RACE" : "race"})


        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        # divide the data based on protected_attribute
        dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

        # Check Default model scores on test data


        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        
        print("---------------RESULTS BEFORE Fair-SSL------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Train model for privileged group

        dataset_orig_male[protected_attribute] = 0
        X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
        clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_male.fit(X_train_male, y_train_male)


        # Train model for unprivileged group

        X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
        clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_female.fit(X_train_female, y_train_female)


        # select fair rows

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        for index,row in dataset_orig_train.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_male = clf_male.predict(row_)
            y_female = clf_female.predict(row_)
            if y_male[0] != y_female[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig_train = dataset_orig_train.drop(index)


        # prepare labeled and unlabeled data

        zero_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        zero_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        one_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        one_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        initial_size = len(dataset_orig_train)*percentage//400

        df1 = zero_zero_df[:initial_size].append(zero_one_df[:initial_size])
        df2 = df1.append(one_zero_df[:initial_size])
        df3 = df2.append(one_one_df[:initial_size])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[initial_size:].append(zero_one_df[initial_size:])
        df2 = df1.append(one_zero_df[initial_size:])
        df3 = df2.append(one_one_df[initial_size:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        # Train self-training model

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        self_training_model = SelfTrainingClassifier(clf)
        self_training_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = self_training_model.predict_proba(X_unl)
        y_pred = self_training_model.predict(X_unl)


        to_keep = []

        for i in range(len(y_pred_proba)):
            if max(y_pred_proba[i]) >= 0.6:
                to_keep.append(i)

        X_unl_certain = X_unl.iloc[to_keep,:]
        y_unl_certain = y_pred[to_keep]

        X_train, y_train = labeled_df.loc[:, labeled_df.columns != 'Probability'], labeled_df['Probability']

        X_train = X_train.append(X_unl_certain)
        y_train = np.concatenate([y_train,y_unl_certain])


        # check scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)

        print("---------------RESULTS AFTER Self-training ------------------")


        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Balance data for further improvement

        X_train['Probability'] = y_train
        dataset_orig_train = X_train

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        zero_zero_to_be_incresed = maximum - zero_zero ## where both are 0
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        df_zero_zero['race'] = df_zero_zero['race'].astype(str)
        df_one_zero['race'] = df_one_zero['race'].astype(str)
        df_one_one['race'] = df_one_one['race'].astype(str)

        df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'MEPS15')
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'MEPS15')
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'MEPS15')


        df = df_zero_zero.append(df_one_zero)
        df = df.append(df_one_one)

        df['race'] = df['race'].astype(float)


        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df = df.append(df_zero_one)


        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train) # LSR

        print("---------------RESULTS AFTER balancing ------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("=============================================================================")


    if dataset == "MEPS16":
        
        ## Load dataset
        dataset_orig = pd.read_csv('../data/MEPS/h192.csv')

        # ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        dataset_orig = dataset_orig.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              'POVCAT16' : 'POVCAT', 'INSCOV16' : 'INSCOV'})


        dataset_orig = dataset_orig[dataset_orig['PANEL'] == 21]
        dataset_orig = dataset_orig[dataset_orig['REGION'] >= 0] # remove values -1
        dataset_orig = dataset_orig[dataset_orig['AGE'] >= 0] # remove values -1
        dataset_orig = dataset_orig[dataset_orig['MARRY'] >= 0] # remove values -1, -7, -8, -9
        dataset_orig = dataset_orig[dataset_orig['ASTHDX'] >= 0] # remove values -1, -7, -8, -9
        dataset_orig = dataset_orig[(dataset_orig[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                 'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]

        # ## Change symbolics to numerics
        dataset_orig['RACEV2X'] = np.where((dataset_orig['HISPANX'] == 2 ) & (dataset_orig['RACEV2X'] == 1), 1, dataset_orig['RACEV2X'])
        dataset_orig['RACEV2X'] = np.where(dataset_orig['RACEV2X'] != 1 , 0, dataset_orig['RACEV2X'])
        dataset_orig = dataset_orig.rename(columns={"RACEV2X" : "RACE"})
        # dataset_orig['UTILIZATION'] = np.where(dataset_orig['UTILIZATION'] >= 10, 1, 0)



        def utilization(row):
                return row['OBTOTV16'] + row['OPTOTV16'] + row['ERTOT16'] + row['IPNGTD16'] + row['HHTOTD16']

        dataset_orig['TOTEXP16'] = dataset_orig.apply(lambda row: utilization(row), axis=1)
        lessE = dataset_orig['TOTEXP16'] < 10.0
        dataset_orig.loc[lessE,'TOTEXP16'] = 0.0
        moreE = dataset_orig['TOTEXP16'] >= 10.0
        dataset_orig.loc[moreE,'TOTEXP16'] = 1.0

        dataset_orig = dataset_orig.rename(columns = {'TOTEXP16' : 'UTILIZATION'})

        dataset_orig = dataset_orig[['REGION','AGE','SEX','RACE','MARRY',
                                         'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                         'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                         'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                         'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                         'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION', 'PERWT16F']]

        dataset_orig = dataset_orig.rename(columns={"UTILIZATION": "Probability","RACE" : "race"})


        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        # divide the data based on protected_attribute
        dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

        # Check Default model scores on test data


        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        
        print("---------------RESULTS BEFORE Fair-SSL------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Train model for privileged group

        dataset_orig_male[protected_attribute] = 0
        X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
        clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_male.fit(X_train_male, y_train_male)


        # Train model for unprivileged group

        X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
        clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_female.fit(X_train_female, y_train_female)


        # select fair rows

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        for index,row in dataset_orig_train.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_male = clf_male.predict(row_)
            y_female = clf_female.predict(row_)
            if y_male[0] != y_female[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig_train = dataset_orig_train.drop(index)


        # prepare labeled and unlabeled data

        zero_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        zero_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        one_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        one_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        initial_size = len(dataset_orig_train)*percentage//400

        df1 = zero_zero_df[:initial_size].append(zero_one_df[:initial_size])
        df2 = df1.append(one_zero_df[:initial_size])
        df3 = df2.append(one_one_df[:initial_size])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[initial_size:].append(zero_one_df[initial_size:])
        df2 = df1.append(one_zero_df[initial_size:])
        df3 = df2.append(one_one_df[initial_size:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        # Train self-training model

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        self_training_model = SelfTrainingClassifier(clf)
        self_training_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = self_training_model.predict_proba(X_unl)
        y_pred = self_training_model.predict(X_unl)


        to_keep = []

        for i in range(len(y_pred_proba)):
            if max(y_pred_proba[i]) >= 0.6:
                to_keep.append(i)

        X_unl_certain = X_unl.iloc[to_keep,:]
        y_unl_certain = y_pred[to_keep]

        X_train, y_train = labeled_df.loc[:, labeled_df.columns != 'Probability'], labeled_df['Probability']

        X_train = X_train.append(X_unl_certain)
        y_train = np.concatenate([y_train,y_unl_certain])


        # check scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)

        print("---------------RESULTS AFTER Self-training ------------------")


        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Balance data for further improvement

        X_train['Probability'] = y_train
        dataset_orig_train = X_train

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        zero_zero_to_be_incresed = maximum - zero_zero ## where both are 0
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        df_zero_zero['race'] = df_zero_zero['race'].astype(str)
        df_one_zero['race'] = df_one_zero['race'].astype(str)
        df_one_one['race'] = df_one_one['race'].astype(str)

        df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'MEPS16')
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'MEPS16')
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'MEPS16')


        df = df_zero_zero.append(df_one_zero)
        df = df.append(df_one_one)

        df['race'] = df['race'].astype(float)


        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df = df.append(df_zero_one)


        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train) # LSR

        print("---------------RESULTS AFTER balancing ------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("=============================================================================")


    if dataset == "Heart":
        
        ## Load dataset
        dataset_orig = pd.read_csv('../data/processed.cleveland.data.csv')

        # ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        ## calculate mean of age column
        mean = dataset_orig.loc[:,"age"].mean()
        dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 0, 1)

        ## Make goal column binary
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] > 0, 1, 0)


        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        # divide the data based on protected_attribute
        dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

        # Check Default model scores on test data


        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        
        print("---------------RESULTS BEFORE Fair-SSL------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Train model for privileged group

        dataset_orig_male[protected_attribute] = 0
        X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
        clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_male.fit(X_train_male, y_train_male)


        # Train model for unprivileged group

        X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
        clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_female.fit(X_train_female, y_train_female)


        # select fair rows

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        for index,row in dataset_orig_train.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_male = clf_male.predict(row_)
            y_female = clf_female.predict(row_)
            if y_male[0] != y_female[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig_train = dataset_orig_train.drop(index)


        # prepare labeled and unlabeled data

        zero_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        zero_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        one_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        one_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        initial_size = len(dataset_orig_train)*percentage//100

        df1 = zero_zero_df[:initial_size].append(zero_one_df[:initial_size])
        df2 = df1.append(one_zero_df[:initial_size])
        df3 = df2.append(one_one_df[:initial_size])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[initial_size:].append(zero_one_df[initial_size:])
        df2 = df1.append(one_zero_df[initial_size:])
        df3 = df2.append(one_one_df[initial_size:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        # Train self-training model

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        self_training_model = SelfTrainingClassifier(clf)
        self_training_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = self_training_model.predict_proba(X_unl)
        y_pred = self_training_model.predict(X_unl)


        to_keep = []

        for i in range(len(y_pred_proba)):
            if max(y_pred_proba[i]) >= 0.6:
                to_keep.append(i)

        X_unl_certain = X_unl.iloc[to_keep,:]
        y_unl_certain = y_pred[to_keep]

        X_train, y_train = labeled_df.loc[:, labeled_df.columns != 'Probability'], labeled_df['Probability']

        X_train = X_train.append(X_unl_certain)
        y_train = np.concatenate([y_train,y_unl_certain])


        # check scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)

        print("---------------RESULTS AFTER Self-training ------------------")


        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Balance data for further improvement

        X_train['Probability'] = y_train
        dataset_orig_train = X_train

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        zero_one_to_be_incresed = maximum - zero_one ## where class is 0 attribute is 1
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        df_zero_one['age'] = df_zero_one['age'].astype(str)
        df_zero_one['sex'] = df_zero_one['sex'].astype(str)


        df_one_zero['age'] = df_one_zero['age'].astype(str)
        df_one_zero['sex'] = df_one_zero['sex'].astype(str)

        df_one_one['age'] = df_one_one['age'].astype(str)
        df_one_one['sex'] = df_one_one['sex'].astype(str)


        df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'Heart')
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'Heart')
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'Heart')


        df = df_zero_one.append(df_one_zero)
        df = df.append(df_one_one)

        df['age'] = df['age'].astype(float)
        df['sex'] = df['sex'].astype(float)

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df = df.append(df_zero_zero)

        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train) # LSR

        print("---------------RESULTS AFTER balancing ------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("=============================================================================")


    if dataset == "Bank":
        
        ## Load dataset
        dataset_orig = pd.read_csv('../data/bank.csv')

        # ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        dataset_orig = dataset_orig.drop(['job','marital','education','contact','month','poutcome'],axis=1)

        dataset_orig['default'] = np.where(dataset_orig['default'] == 'no', 0, 1)
        dataset_orig['housing'] = np.where(dataset_orig['housing'] == 'no', 0, 1)
        dataset_orig['loan'] = np.where(dataset_orig['loan'] == 'no', 0, 1)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 'yes', 1, 0)
        dataset_orig['age'] = np.where(dataset_orig['age'] >= 30, 1, 0)


        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        # divide the data based on protected_attribute
        dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

        # Check Default model scores on test data


        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        
        print("---------------RESULTS BEFORE Fair-SSL------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Train model for privileged group

        dataset_orig_male[protected_attribute] = 0
        X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
        clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_male.fit(X_train_male, y_train_male)


        # Train model for unprivileged group

        X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
        clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_female.fit(X_train_female, y_train_female)


        # select fair rows

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        for index,row in dataset_orig_train.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_male = clf_male.predict(row_)
            y_female = clf_female.predict(row_)
            if y_male[0] != y_female[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig_train = dataset_orig_train.drop(index)


        # prepare labeled and unlabeled data

        zero_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        zero_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        one_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        one_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        initial_size = len(dataset_orig_train)*percentage//400

        df1 = zero_zero_df[:initial_size].append(zero_one_df[:initial_size])
        df2 = df1.append(one_zero_df[:initial_size])
        df3 = df2.append(one_one_df[:initial_size])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[initial_size:].append(zero_one_df[initial_size:])
        df2 = df1.append(one_zero_df[initial_size:])
        df3 = df2.append(one_one_df[initial_size:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        # Train self-training model

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        self_training_model = SelfTrainingClassifier(clf)
        self_training_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = self_training_model.predict_proba(X_unl)
        y_pred = self_training_model.predict(X_unl)


        to_keep = []

        for i in range(len(y_pred_proba)):
            if max(y_pred_proba[i]) >= 0.6:
                to_keep.append(i)

        X_unl_certain = X_unl.iloc[to_keep,:]
        y_unl_certain = y_pred[to_keep]

        X_train, y_train = labeled_df.loc[:, labeled_df.columns != 'Probability'], labeled_df['Probability']

        X_train = X_train.append(X_unl_certain)
        y_train = np.concatenate([y_train,y_unl_certain])


        # check scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)

        print("---------------RESULTS AFTER Self-training ------------------")


        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Balance data for further improvement

        X_train['Probability'] = y_train
        dataset_orig_train = X_train

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        zero_zero_to_be_incresed = maximum - zero_zero ## where both are 0
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        df_zero_zero['age'] = df_zero_zero['age'].astype(str)
        df_one_zero['age'] = df_one_zero['age'].astype(str)
        df_one_one['age'] = df_one_one['age'].astype(str)


        df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'Bank')
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'Bank')
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'Bank')


        df = df_zero_zero.append(df_one_zero)
        df = df.append(df_one_one)

        df['age'] = df['age'].astype(float)

        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df = df.append(df_zero_one)

        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train) # LSR

        print("---------------RESULTS AFTER balancing ------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("=============================================================================")



    if dataset == "Student":
        
        ## Load dataset
        dataset_orig = pd.read_csv('../data/Student.csv')

        # ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        ## Drop categorical features
        dataset_orig = dataset_orig.drop(['school','address', 'famsize', 'Pstatus','Mjob', 'Fjob', 'reason', 'guardian'],axis=1)

        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'M', 1, 0)
        dataset_orig['schoolsup'] = np.where(dataset_orig['schoolsup'] == 'yes', 1, 0)
        dataset_orig['famsup'] = np.where(dataset_orig['famsup'] == 'yes', 1, 0)
        dataset_orig['paid'] = np.where(dataset_orig['paid'] == 'yes', 1, 0)
        dataset_orig['activities'] = np.where(dataset_orig['activities'] == 'yes', 1, 0)
        dataset_orig['nursery'] = np.where(dataset_orig['nursery'] == 'yes', 1, 0)
        dataset_orig['higher'] = np.where(dataset_orig['higher'] == 'yes', 1, 0)
        dataset_orig['internet'] = np.where(dataset_orig['internet'] == 'yes', 1, 0)
        dataset_orig['romantic'] = np.where(dataset_orig['romantic'] == 'yes', 1, 0)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] > 12, 1, 0)


        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        # divide the data based on protected_attribute
        dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

        # Check Default model scores on test data


        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        
        print("---------------RESULTS BEFORE Fair-SSL------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Train model for privileged group

        dataset_orig_male[protected_attribute] = 0
        X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
        clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_male.fit(X_train_male, y_train_male)


        # Train model for unprivileged group

        X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
        clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_female.fit(X_train_female, y_train_female)


        # select fair rows

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        for index,row in dataset_orig_train.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_male = clf_male.predict(row_)
            y_female = clf_female.predict(row_)
            if y_male[0] != y_female[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig_train = dataset_orig_train.drop(index)


        # prepare labeled and unlabeled data

        zero_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        zero_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        one_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        one_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        initial_size = len(dataset_orig_train)*percentage//400

        df1 = zero_zero_df[:initial_size].append(zero_one_df[:initial_size])
        df2 = df1.append(one_zero_df[:initial_size])
        df3 = df2.append(one_one_df[:initial_size])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[initial_size:].append(zero_one_df[initial_size:])
        df2 = df1.append(one_zero_df[initial_size:])
        df3 = df2.append(one_one_df[initial_size:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        # Train self-training model

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        self_training_model = SelfTrainingClassifier(clf)
        self_training_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = self_training_model.predict_proba(X_unl)
        y_pred = self_training_model.predict(X_unl)


        to_keep = []

        for i in range(len(y_pred_proba)):
            if max(y_pred_proba[i]) >= 0.6:
                to_keep.append(i)

        X_unl_certain = X_unl.iloc[to_keep,:]
        y_unl_certain = y_pred[to_keep]

        X_train, y_train = labeled_df.loc[:, labeled_df.columns != 'Probability'], labeled_df['Probability']

        X_train = X_train.append(X_unl_certain)
        y_train = np.concatenate([y_train,y_unl_certain])


        # check scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)

        print("---------------RESULTS AFTER Self-training ------------------")


        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Balance data for further improvement

        X_train['Probability'] = y_train
        dataset_orig_train = X_train

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        zero_one_to_be_incresed = maximum - zero_one ## where class is 0 attribute is 1
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]


        df_zero_one['sex'] = df_zero_one['sex'].astype(str)
        df_one_zero['sex'] = df_one_zero['sex'].astype(str)
        df_one_one['sex'] = df_one_one['sex'].astype(str)


        df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'Student')
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'Student')
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'Student')


        df = df_zero_one.append(df_one_zero)
        df = df.append(df_one_one)

        df['sex'] = df['sex'].astype(float)

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df = df.append(df_zero_zero)

        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train) # LSR

        print("---------------RESULTS AFTER balancing ------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("=============================================================================")


    if dataset == "Default":
        
        ## Load dataset
        dataset_orig = pd.read_csv('../data/default_of_credit_card_clients_first_row_removed.csv')

        # ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 2, 0,1)


        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        # divide the data based on protected_attribute
        dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

        # Check Default model scores on test data


        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        
        print("---------------RESULTS BEFORE Fair-SSL------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Train model for privileged group

        dataset_orig_male[protected_attribute] = 0
        X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
        clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_male.fit(X_train_male, y_train_male)


        # Train model for unprivileged group

        X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
        clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_female.fit(X_train_female, y_train_female)


        # select fair rows

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        for index,row in dataset_orig_train.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_male = clf_male.predict(row_)
            y_female = clf_female.predict(row_)
            if y_male[0] != y_female[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig_train = dataset_orig_train.drop(index)


        # prepare labeled and unlabeled data

        zero_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        zero_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        one_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        one_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        initial_size = len(dataset_orig_train)*percentage//400

        df1 = zero_zero_df[:initial_size].append(zero_one_df[:initial_size])
        df2 = df1.append(one_zero_df[:initial_size])
        df3 = df2.append(one_one_df[:initial_size])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[initial_size:].append(zero_one_df[initial_size:])
        df2 = df1.append(one_zero_df[initial_size:])
        df3 = df2.append(one_one_df[initial_size:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        # Train self-training model

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        self_training_model = SelfTrainingClassifier(clf)
        self_training_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = self_training_model.predict_proba(X_unl)
        y_pred = self_training_model.predict(X_unl)


        to_keep = []

        for i in range(len(y_pred_proba)):
            if max(y_pred_proba[i]) >= 0.6:
                to_keep.append(i)

        X_unl_certain = X_unl.iloc[to_keep,:]
        y_unl_certain = y_pred[to_keep]

        X_train, y_train = labeled_df.loc[:, labeled_df.columns != 'Probability'], labeled_df['Probability']

        X_train = X_train.append(X_unl_certain)
        y_train = np.concatenate([y_train,y_unl_certain])


        # check scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)

        print("---------------RESULTS AFTER Self-training ------------------")


        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Balance data for further improvement

        X_train['Probability'] = y_train
        dataset_orig_train = X_train

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        zero_one_to_be_incresed = maximum - zero_one ## where class is 0 attribute is 1
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]


        df_zero_one['sex'] = df_zero_one['sex'].astype(str)
        df_one_zero['sex'] = df_one_zero['sex'].astype(str)
        df_one_one['sex'] = df_one_one['sex'].astype(str)


        df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'Default')
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'Default')
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'Default')


        df = df_zero_one.append(df_one_zero)
        df = df.append(df_one_one)

        df['sex'] = df['sex'].astype(float)

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df = df.append(df_zero_zero)

        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train) # LSR

        print("---------------RESULTS AFTER balancing ------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("=============================================================================")



    if dataset == "Home-Credit":
        
        ## Load dataset
        dataset_orig = pd.read_csv('../data/default_of_credit_card_clients_first_row_removed.csv')

        # ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 2, 0,1)


        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        # divide the data based on protected_attribute
        dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

        # Check Default model scores on test data


        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        
        print("---------------RESULTS BEFORE Fair-SSL------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Train model for privileged group

        dataset_orig_male[protected_attribute] = 0
        X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
        clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_male.fit(X_train_male, y_train_male)


        # Train model for unprivileged group

        X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
        clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_female.fit(X_train_female, y_train_female)


        # select fair rows

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        for index,row in dataset_orig_train.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_male = clf_male.predict(row_)
            y_female = clf_female.predict(row_)
            if y_male[0] != y_female[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig_train = dataset_orig_train.drop(index)


        # prepare labeled and unlabeled data

        zero_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        zero_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        one_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        one_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        initial_size = len(dataset_orig_train)*percentage//400

        df1 = zero_zero_df[:initial_size].append(zero_one_df[:initial_size])
        df2 = df1.append(one_zero_df[:initial_size])
        df3 = df2.append(one_one_df[:initial_size])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[initial_size:].append(zero_one_df[initial_size:])
        df2 = df1.append(one_zero_df[initial_size:])
        df3 = df2.append(one_one_df[initial_size:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        # Train self-training model

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        self_training_model = SelfTrainingClassifier(clf)
        self_training_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = self_training_model.predict_proba(X_unl)
        y_pred = self_training_model.predict(X_unl)


        to_keep = []

        for i in range(len(y_pred_proba)):
            if max(y_pred_proba[i]) >= 0.6:
                to_keep.append(i)

        X_unl_certain = X_unl.iloc[to_keep,:]
        y_unl_certain = y_pred[to_keep]

        X_train, y_train = labeled_df.loc[:, labeled_df.columns != 'Probability'], labeled_df['Probability']

        X_train = X_train.append(X_unl_certain)
        y_train = np.concatenate([y_train,y_unl_certain])


        # check scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)

        print("---------------RESULTS AFTER Self-training ------------------")


        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Balance data for further improvement

        X_train['Probability'] = y_train
        dataset_orig_train = X_train

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        zero_one_to_be_incresed = maximum - zero_one ## where class is 0 attribute is 1
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]


        df_zero_one['sex'] = df_zero_one['sex'].astype(str)
        df_one_zero['sex'] = df_one_zero['sex'].astype(str)
        df_one_one['sex'] = df_one_one['sex'].astype(str)


        df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'Default')
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'Default')
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'Default')


        df = df_zero_one.append(df_one_zero)
        df = df.append(df_one_one)

        df['sex'] = df['sex'].astype(float)

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df = df.append(df_zero_zero)

        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train) # LSR

        print("---------------RESULTS AFTER balancing ------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("=============================================================================")



    if dataset == "German":
        
        ## Load dataset
        dataset_orig = pd.read_csv('../data/GermanData.csv')

        # ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        dataset_orig = dataset_orig.drop(['1','2','4','5','8','10','11','12','14','15','16','17','18','19','20'],axis=1)

        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A91', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A92', 0, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A93', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A94', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A95', 0, dataset_orig['sex'])

        # mean = dataset_orig.loc[:,"age"].mean()
        # dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
        dataset_orig['age'] = np.where(dataset_orig['age'] >= 25, 1, 0)
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A30', 1, dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A31', 1, dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A32', 1, dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A33', 2, dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A34', 3, dataset_orig['credit_history'])

        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A61', 1, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A62', 1, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A63', 2, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A64', 2, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A65', 3, dataset_orig['savings'])

        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A72', 1, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A73', 1, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A74', 2, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A75', 2, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A71', 3, dataset_orig['employment'])



        ## ADD Columns
        dataset_orig['credit_history=Delay'] = 0
        dataset_orig['credit_history=None/Paid'] = 0
        dataset_orig['credit_history=Other'] = 0

        dataset_orig['credit_history=Delay'] = np.where(dataset_orig['credit_history'] == 1, 1, dataset_orig['credit_history=Delay'])
        dataset_orig['credit_history=None/Paid'] = np.where(dataset_orig['credit_history'] == 2, 1, dataset_orig['credit_history=None/Paid'])
        dataset_orig['credit_history=Other'] = np.where(dataset_orig['credit_history'] == 3, 1, dataset_orig['credit_history=Other'])

        dataset_orig['savings=500+'] = 0
        dataset_orig['savings=<500'] = 0
        dataset_orig['savings=Unknown/None'] = 0

        dataset_orig['savings=500+'] = np.where(dataset_orig['savings'] == 1, 1, dataset_orig['savings=500+'])
        dataset_orig['savings=<500'] = np.where(dataset_orig['savings'] == 2, 1, dataset_orig['savings=<500'])
        dataset_orig['savings=Unknown/None'] = np.where(dataset_orig['savings'] == 3, 1, dataset_orig['savings=Unknown/None'])

        dataset_orig['employment=1-4 years'] = 0
        dataset_orig['employment=4+ years'] = 0
        dataset_orig['employment=Unemployed'] = 0

        dataset_orig['employment=1-4 years'] = np.where(dataset_orig['employment'] == 1, 1, dataset_orig['employment=1-4 years'])
        dataset_orig['employment=4+ years'] = np.where(dataset_orig['employment'] == 2, 1, dataset_orig['employment=4+ years'])
        dataset_orig['employment=Unemployed'] = np.where(dataset_orig['employment'] == 3, 1, dataset_orig['employment=Unemployed'])


        dataset_orig = dataset_orig.drop(['credit_history','savings','employment'],axis=1)
        ## In dataset 1 means good, 2 means bad for probability. I change 2 to 0
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 2, 0, 1)

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        # divide the data based on protected_attribute
        dataset_orig_male , dataset_orig_female = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

        # Check Default model scores on test data


        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        
        print("---------------RESULTS BEFORE Fair-SSL------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Train model for privileged group

        dataset_orig_male[protected_attribute] = 0
        X_train_male, y_train_male = dataset_orig_male.loc[:, dataset_orig_male.columns != 'Probability'], dataset_orig_male['Probability']
        clf_male = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_male.fit(X_train_male, y_train_male)


        # Train model for unprivileged group

        X_train_female, y_train_female = dataset_orig_female.loc[:, dataset_orig_female.columns != 'Probability'], dataset_orig_female['Probability']
        clf_female = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf_female.fit(X_train_female, y_train_female)


        # select fair rows

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        for index,row in dataset_orig_train.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_male = clf_male.predict(row_)
            y_female = clf_female.predict(row_)
            if y_male[0] != y_female[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig_train = dataset_orig_train.drop(index)


        # prepare labeled and unlabeled data

        zero_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        zero_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        one_zero_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        one_one_df = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        initial_size = len(dataset_orig_train)*percentage//400

        df1 = zero_zero_df[:initial_size].append(zero_one_df[:initial_size])
        df2 = df1.append(one_zero_df[:initial_size])
        df3 = df2.append(one_one_df[:initial_size])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[initial_size:].append(zero_one_df[initial_size:])
        df2 = df1.append(one_zero_df[initial_size:])
        df3 = df2.append(one_one_df[initial_size:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        # Train self-training model

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)
        self_training_model = SelfTrainingClassifier(clf)
        self_training_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = self_training_model.predict_proba(X_unl)
        y_pred = self_training_model.predict(X_unl)


        to_keep = []

        for i in range(len(y_pred_proba)):
            if max(y_pred_proba[i]) >= 0.6:
                to_keep.append(i)

        X_unl_certain = X_unl.iloc[to_keep,:]
        y_unl_certain = y_pred[to_keep]

        X_train, y_train = labeled_df.loc[:, labeled_df.columns != 'Probability'], labeled_df['Probability']

        X_train = X_train.append(X_unl_certain)
        y_train = np.concatenate([y_train,y_unl_certain])


        # check scores on test data

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train)

        print("---------------RESULTS AFTER Self-training ------------------")


        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        # Balance data for further improvement

        X_train['Probability'] = y_train
        dataset_orig_train = X_train

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        zero_zero_to_be_incresed = maximum - zero_zero ## where both are 0
        zero_one_to_be_incresed = maximum - zero_one ## where class is 0 attribute is 1
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]


        df_zero_zero['sex'] = df_zero_zero['sex'].astype(str)
        df_zero_one['sex'] = df_zero_one['sex'].astype(str)
        df_one_zero['sex'] = df_one_zero['sex'].astype(str)


        df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'German')
        df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'German')
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'German')


        df = df_zero_zero.append(df_zero_one)
        df = df.append(df_one_zero)

        df['sex'] = df['sex'].astype(float)

        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]
        df = df.append(df_one_one)

        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train,y_train) # LSR

        print("---------------RESULTS AFTER balancing ------------------")

        print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
        print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
        print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
        print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
        print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
        print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

        print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
        print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("=============================================================================")



if __name__ == "__main__":

    datasets = ["adult","compas","German","MEPS15","MEPS16","Heart","Bank","Student","Default","Home-Credit"]
    protected_attributes = ["sex","sex","sex","race","race","age","age","sex","sex","sex"]
    percentage = 10 # this value needs to be changed for different size of initial set (for example 1,5,10,20)

    for each in range(len(datasets)):

        self_training(datasets[each],protected_attributes[each],percentage)
