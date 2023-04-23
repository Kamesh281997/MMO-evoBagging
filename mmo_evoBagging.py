
import sklearn
from sklearn.metrics import *
import numpy as np
import pandas as pd
import random
import math
from scipy import stats
import copy
from multiprocessing import Pool
from functools import partial
import time
from sklearn.metrics import f1_score, precision_score, roc_auc_score, recall_score
from imblearn.metrics import specificity_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import f1_score
import xgboost as xgb
from diversity_1 import *
from imblearn.metrics import geometric_mean_score
class MMO_EvoBagging:
    def __init__(self, X_train, y_train, X_test, y_test, n_bags,
                n_select, n_new_bags, 
                max_initial_size, n_crossover, 
                n_mutation, mutation_size, 
                size_coef, clf_coef, metric, procs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_bags = n_bags
        self.n_select = n_select
        self.n_new_bags = n_new_bags
        self.max_initial_size = max_initial_size
        self.n_crossover = n_crossover
        self.n_mutation = n_mutation
        self.mutation_size = mutation_size
        self.size_coef = size_coef
        self.clf_coef = clf_coef
        self.metric = metric
        self.procs = procs

    def get_metrics(y_true, y_pred):
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        return f1, precision, recall, auc
    
    def get_diversity_measures(self, preds, y, bags):
        q_statistics(preds, y, bags)
        return 

    def get_score(self, X, y):
        estimator = []
        pipeline_dt = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy'))])
        parameters_dt = {'clf__max_depth': list(range(1,X.shape[1]))}
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average = 'macro')
        grid_search_dt = GridSearchCV(pipeline_dt, parameters_dt, cv=5, n_jobs=-1, scoring=scorer)
      
        scorer1 = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average = 'macro')
        parameters_svm = {'kernel':['rbf','sigmoid'], 'gamma':np.logspace(0, 5, num=5, base=2.0)}
        grid_search_svm = GridSearchCV(svm.SVC(probability=True), parameters_svm, error_score='raise',cv=5, n_jobs=-1, scoring=scorer1)
       
        estimator.append(('DTC', grid_search_dt))
        estimator.append(('SVM', grid_search_svm))
        clf = VotingClassifier(estimators = estimator, voting ='soft')
        clf.fit(X, y)
        preds = clf.predict(X)
        perf = eval(f"{self.metric}_score(y, preds)")
        return perf, clf, preds
        
    def get_payoff(self, bags, idx): 
        met, clf, preds = self.get_score(bags[idx]['X'], bags[idx]['y'].values.ravel())
        payoff = ((self.size_coef+(bags[idx]['X'].shape[0] ))/self.size_coef)
        bags[idx]['clf'] = clf
        bags[idx]['metric'] = met
        bags[idx]['preds'] = preds
        bags[idx]['payoff'] = payoff
        bags[idx]['size'] = bags[idx]['X'].shape[0]
        return idx, copy.deepcopy(bags[idx])
    
    def get_diversity( self, bags, idx):
        
        evobagging_preds = np.zeros((self.y_train.shape[0],100))
        for i, bag in bags.items():
            preds = bag['clf'].predict(self.X_train)
            evobagging_preds[:, i] = preds
       
        for model in ["evobagging"]:
            self.get_diversity_measures(eval(f"{model}_preds"), self.y_train.to_numpy(), bags)
       
        for i, bag in bags.items():
            diverse=bag['diverse']
            payoff=bag['payoff']
            met=bag['metric'] 
            met1= (((float)((self.clf_coef)+(met*100 )))/(float)(self.clf_coef))
            diversity=met1*(payoff+diverse)
            bag['payoff']=diversity
        
        return idx, copy.deepcopy(bags[idx])
    

    def naive_selection(self, bags, mode="selection"):
        selected_bag_dict = {}
        selected_ids = []
        bag_idx, payoff_list = [], []
        for idx, bag in bags.items():
            bag_idx.append(idx)
            payoff_list.append(bag['payoff'])
        if mode=="selection":
            selected_ids = [idx for _, idx in sorted(zip(payoff_list, bag_idx), reverse=True)][:self.n_select]
            selected_bag_dict = {i: bags[i] for i in selected_ids}
            return selected_bag_dict, selected_ids
        elif mode=="crossover":
            selected_ids = [idx for _, idx in sorted(zip(payoff_list, bag_idx), reverse=True)][:self.n_crossover]
            return None, selected_ids
        elif mode=="generation":
            selected_ids = [idx for _, idx in sorted(zip(payoff_list, bag_idx))][:self.n_new_bags]
            return None, selected_ids


    def gen_new_bag(self):
        initial_size = random.randrange(int(self.max_initial_size/2), self.max_initial_size)
        bag_idx = random.choices(list(self.y_train.index), k=initial_size)
        temp_X = self.X_train.loc[bag_idx, :]
        temp_y = self.y_train.loc[bag_idx, :]
        return {'X': temp_X, 'y': temp_y}

    def generation_gap(self, new_bags, bags):
        glist=[]
        _, generation_idx = self.naive_selection(bags, mode="generation")
        for _ in range(self.n_new_bags):
            new_bag = self.gen_new_bag()
            new_bag_idx = random.choice(list(generation_idx))
            generation_idx.remove(new_bag_idx)
            new_bags[new_bag_idx] = new_bag
        return new_bags

    def crossover_pair(self, parent1, parent2):    
        preds_1 = parent1['preds']
        wrong_idx_1 = preds_1 != parent1['y'][0]
        parent1_leave_idx = parent1['X'].index[wrong_idx_1]
        preds_2 = parent2['preds']
        wrong_idx_2 = preds_2 != parent2['y'][0]
        parent2_leave_idx = parent2['X'].index[wrong_idx_2]
        new_parent1_X = parent1['X'].loc[~parent1['X'].index.isin(parent1_leave_idx)]
        leave_parent1_X = parent1['X'].loc[parent1['X'].index.isin(parent1_leave_idx)]
        new_parent1_y = parent1['y'].loc[~parent1['y'].index.isin(parent1_leave_idx)]
        leave_parent1_y = parent1['y'].loc[parent1['y'].index.isin(parent1_leave_idx)]
        new_parent2_X = parent2['X'].loc[~parent2['X'].index.isin(parent2_leave_idx)]
        leave_parent2_X = parent2['X'].loc[parent2['X'].index.isin(parent2_leave_idx)]
        new_parent2_y = parent2['y'].loc[~parent2['y'].index.isin(parent2_leave_idx)]
        leave_parent2_y = parent2['y'].loc[parent2['y'].index.isin(parent2_leave_idx)]  
        child1, child2 = {}, {}
        child1['X'] = pd.concat([new_parent1_X, leave_parent2_X])
        child1['y'] = pd.concat([new_parent1_y, leave_parent2_y])
        child2['X'] = pd.concat([new_parent2_X, leave_parent1_X])
        child2['y'] = pd.concat([new_parent2_y, leave_parent1_y])

        return child1, child2

    def crossover(self, new_bags, bags):
        _, crossover_pool_idx = self.naive_selection(bags, mode="crossover")
        remaining_idx = list(set(range(len(bags))) - set(new_bags.keys()))
        random.shuffle(remaining_idx)
        for j in range(0, self.n_crossover, 2):
            parent1 = bags[crossover_pool_idx[j]]
            parent2 = bags[crossover_pool_idx[j + 1]]
            child1, child2 = self.crossover_pair(parent1, parent2)
            new_bags[remaining_idx[j]] = child1
            new_bags[remaining_idx[j + 1]] = child2
        return new_bags
    
    def mutation(self, bags):
        bag_mutation_idx = random.sample(list(bags.keys()), k=self.n_mutation)
        for j in bag_mutation_idx:
            bag_idx = bags[j]['y'].index
            leftover_idx = list(set(self.X_train.index) - set(bag_idx))
            leave_idx = random.sample(list(bag_idx), k=self.mutation_size)
            new_idx = random.choices(list(leftover_idx), k=self.mutation_size)
            keep_bag_X = bags[j]['X'].loc[~bag_idx.isin(leave_idx)]
            keep_bag_y = bags[j]['y'].loc[~bag_idx.isin(leave_idx)]
            new_bag_X = self.X_train.loc[new_idx]
            new_bag_y = self.y_train.loc[new_idx]
            bags[j]['X'] = pd.concat([keep_bag_X, new_bag_X])
            bags[j]['y'] = pd.concat([keep_bag_y, new_bag_y])
        return bags, bag_mutation_idx

    def evaluate_bags(self, bags):
        with Pool(self.procs) as p:
            output = p.map(partial(self.get_payoff, bags), list(bags.keys()))     
        bags = {idx: bag for (idx, bag) in output}
        return bags
    
    def re_evaluate_bags(self, bags, X_test, y_test):
        with Pool(self.procs) as p: 
            output = p.map(partial(self.get_diversity, bags), list(bags.keys())) 
        bags = {idx: bag for (idx, bag) in output}
        return bags

    def voting_metric(self, X, y, bags, return_preds=False):
        if return_preds:
            preds_list = []
            for bag in bags.values():
                bag_preds = bag['clf'].predict(X)
                preds_list.append(bag_preds)
            temp_preds = np.stack(preds_list)
            final_preds = stats.mode(temp_preds).mode[0]
            met = eval(f"{self.metric}_score(y.loc[:, 0], final_preds)")*100
            return met, final_preds
        else:
            preds_list = []
            for bag in bags.values():
                bag_preds = bag['clf'].predict(X)
                preds_list.append(bag_preds)
            temp_preds = np.stack(preds_list)
            final_preds = stats.mode(temp_preds).mode[0]
            met = eval(f"{self.metric}_score(y.loc[:, 0], final_preds)")*100
            return met
    
    def voting_metric_roc(self, bags, X, y):
        preds_list = []
        train_preds_list = []
        probs = np.zeros((len(y),))
        for bag in bags.values():
            bag['clf']=bag['clf'].fit(bag['X'],bag['y'].values.ravel())
            bag_preds = bag['clf'].predict(X)
            bag_train_preds = bag['clf'].predict(self.X_train)
            probs += bag['clf'].predict_proba(X)[:, 1]
            preds_list.append(bag_preds)
            train_preds_list.append(bag_train_preds)
        temp_preds = np.stack(preds_list) 
        temp_train_preds = np.stack(train_preds_list)
        final_preds = stats.mode(temp_preds).mode[0]
        final_train_preds = stats.mode(temp_train_preds).mode[0]
        met = eval(f"{self.metric}_score(y, final_preds)")
        probs = probs/len(bags)
        fpr, tpr, _ = roc_curve(y, probs)
        rc_sc=roc_auc_score(y,probs)
        f1 = f1_score(self.y_train, final_train_preds)
        precision = precision_score(self.y_train, final_train_preds)
        recall = recall_score(self.y_train, final_train_preds)
        gm=geometric_mean_score(self.y_train, final_train_preds, average='macro')
        sp=specificity_score(self.y_train, final_train_preds, average='macro')
        print("TTraining F1:",f1)
        print("Training Precision:",precision)
        print("Training Recall:",recall)
        print("Training GMean:",gm)
        print("Training Specificity:",sp)
        f1 = f1_score(y, final_preds)
        precision = precision_score(y, final_preds)
        recall = recall_score(y,final_preds)
        gm=geometric_mean_score(y,final_preds, average='macro')
        sp=specificity_score(y,final_preds, average='macro')
        print("Test F1:",f1)
        print("Test Precision:",precision)
        print("Test Recall:",recall)
        print("Test GMean:",gm)
        print("Training Specificity:",sp)
        return met, fpr, tpr, rc_sc

    def mmo_evobagging_optimization(self, bags, X_test, y_test):
        # selection
        new_bags, _ = self.naive_selection(bags)
        # generation gap
        new_bags = self.generation_gap(new_bags, bags)
        # crossover
        new_bags = self.crossover(new_bags, bags)
        # mutation
        new_bags, _ = self.mutation(new_bags)
        # update population
        bags = copy.deepcopy(new_bags)
        # evaluate
        bags = self.evaluate_bags(bags)
        
        bags = self.re_evaluate_bags(bags, X_test, y_test)
    
        return bags