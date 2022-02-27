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
from sklearn.semi_supervised import LabelPropagation


import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.abspath('..'))

from Measure import measure_final_score,calculate_recall,calculate_far,calculate_precision,calculate_accuracy
from Generate_Samples import generate_samples

def labelpropagation(dataset,protected_attribute):

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

        df1 = zero_zero_df[:1000].append(zero_one_df[:1000])
        df2 = df1.append(one_zero_df[:1000])
        df3 = df2.append(one_one_df[:1000])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[1000:].append(zero_one_df[1000:])
        df2 = df1.append(one_zero_df[1000:])
        df3 = df2.append(one_one_df[1000:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        # Train LabelPropagation model

        label_prop_model = LabelPropagation()
        label_prop_model.fit(X_train, y_train)

        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = label_prop_model.predict_proba(X_unl)
        y_pred = label_prop_model.predict(X_unl)

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

        print("---------------RESULTS AFTER LabelPropagation ------------------")


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

        x = 500


        df1 = zero_zero_df[:x].append(zero_one_df[:x])
        df2 = df1.append(one_zero_df[:x])
        df3 = df2.append(one_one_df[:x])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)


        df1 = zero_zero_df[x:].append(zero_one_df[x:])
        df2 = df1.append(one_zero_df[x:])
        df3 = df2.append(one_one_df[x:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)


        unlabeled_df['Probability'] = -1 ## For sklearn label propagation
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != 'Probability'], mixed_df['Probability']

        label_prop_model = LabelPropagation()
        label_prop_model.fit(X_train, y_train)


        X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != 'Probability'], unlabeled_df['Probability']

        y_pred_proba = label_prop_model.predict_proba(X_unl)
        y_pred = label_prop_model.predict(X_unl)

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

        print("---------------RESULTS AFTER LabelPropagation ------------------")


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



if __name__ == "__main__":

    datasets = ["adult","adult","compas","compas"]
    protected_attributes = ["race","sex","race","sex"]

    for each in range(len(datasets)):

        labelpropagation(datasets[each],protected_attributes[each])