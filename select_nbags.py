from audioop import cross
from cgi import test
from preprocess_data import load_data
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn.metrics import *
import numpy as np
import pandas as pd
import random
from scipy import stats
import copy
from multiprocessing import Pool
from functools import partial
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import f1_score
import xgboost as xgb
from diversity_1 import *

dataset_names = ["pima","dermatalogy","heart_cleveland","hepatitis","thyroid"]
n_bag_range = list(range(9, 39, 3))

for dataset_name in dataset_names:
    print(dataset_name)
    X_train, _, y_train, _ = load_data(dataset_name, test_size=0)
    X_train = np.asarray(X_train)
    y_train = np.squeeze(np.asarray(y_train))
    cv_scores = []
    cv_scores1 = []
    cls=DecisionTreeClassifier()
    estimators = []
    pipeline_dt = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy'))])
    parameters_dt = {'clf__max_depth': list(range(1,X_train.shape[1]))}
    scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average = 'macro')
    grid_search_dt = GridSearchCV(pipeline_dt, parameters_dt,error_score='raise', cv=5, n_jobs=-1, scoring=scorer)   
    parameters_svm = {'kernel':['rbf','sigmoid'], 'gamma':np.logspace(0, 5, num=5, base=2.0)}
    grid_search_svm = GridSearchCV(svm.SVC(probability=True), parameters_svm, error_score='raise',cv=5, n_jobs=-1)  
    estimators.append(('DTC', grid_search_dt))
    estimators.append(('SVM', grid_search_svm))
    for n_bags in tqdm(n_bag_range):
        clf = BaggingClassifier(estimator = cls ,n_estimators=n_bags)
        score = cross_val_score(clf, X_train, y_train, cv=5).mean()
        print("N_Bags::",n_bags)
        print("Score::",score)
        cv_scores.append(score)
    n_bag = n_bag_range[np.argmax(cv_scores)]
    print(f"Max score: {max(cv_scores)}")
    print(f"n_bag: {n_bag}")
    print(f"90-quantile score: {np.quantile(cv_scores, 0.9)}")

    print("=======================================")