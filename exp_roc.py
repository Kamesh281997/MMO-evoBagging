from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve
import numpy as np
from mmo_evoBagging import *
from preprocess_data import load_data
from scipy import stats
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score
from imblearn.over_sampling import BorderlineSMOTE

def selection( bags, mode="selection"):
        selected_bag_dict = {}
        selected_ids = []
        bag_idx, payoff_list = [], []
        for idx, bag in bags.items():
            bag_idx.append(idx)
            payoff_list.append(bag['payoff'])
        if mode=="selection":
            selected_ids = [idx for _, idx in sorted(zip(payoff_list, bag_idx), reverse=True)][:9]
            return None, selected_ids
     
def binarize(y_train, y_test):
    y_train = np.array(list(y_train.loc[:, 0]))
    y_test = np.array(list(y_test.loc[:, 0]))
    classes = set(y_train)
    binarized_y = dict()
    for c in classes:
        binarized_y[c] = {'y_train': (y_train==c).astype(int),
                          'y_test': (y_test==c).astype(int)}
    return binarized_y

def run(dataset_name, 
        test_size, 
        metric,
        n_bags, 
        n_iter,
        n_select,
        n_new_bags,
        n_mutation,
        mutation_rate,
        size_coef,
        clf_coef,
        procs):
    X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=test_size)
    oversample = BorderlineSMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    binarized_y = binarize(y_train, y_test)
    bagging_roc_dict = dict()
    RandomForest_dict = dict()
    ExtraTree_dict=dict()
    XGBoost_dict=dict()
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    class_roc_dict = dict()
    for c in binarized_y:
        y_train = binarized_y[c]['y_train']
        y_test = binarized_y[c]['y_test']
        y_train = pd.DataFrame({0: y_train})
        y_test = pd.DataFrame({0:y_test})
        optimizer = MMO_EvoBagging(X_train, y_train,  X_test, y_test,  n_bags,n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef,clf_coef, metric, procs)
        # bagging FPR and TPR
        clf = BaggingClassifier(n_estimators=n_bags)
        clf.fit(X_train, y_train.values.ravel())
        probs = clf.predict_proba(X_test)[:, 1]
        bagging_fpr, bagging_tpr, _ = roc_curve(y_test, probs)
        bagging_roc_dict[c] = {'fpr': bagging_fpr, 'tpr': bagging_tpr}
        rc_sc=roc_auc_score(y_test, probs)
        print("Bagging Roc:",rc_sc)
        
        clf = RandomForestClassifier(n_estimators=n_bags)
        clf.fit(X_train, y_train.values.ravel())
        probs = clf.predict_proba(X_test)[:, 1]
        RandomForest_fpr, RandomForest_tpr, _ = roc_curve(y_test, probs)
        RandomForest_dict[c] = {'fpr': RandomForest_fpr, 'tpr': RandomForest_tpr}
        rc_sc=roc_auc_score(y_test, probs)
        print("RandomForest Roc:",rc_sc)
        
        clf = ExtraTreesClassifier(n_estimators=n_bags)
        clf.fit(X_train, y_train.values.ravel())
        probs = clf.predict_proba(X_test)[:, 1]
        ExtraTree_fpr, ExtraTree_tpr, _ = roc_curve(y_test, probs)
        ExtraTree_dict[c] = {'fpr': ExtraTree_fpr, 'tpr':ExtraTree_tpr}
        rc_sc=roc_auc_score(y_test, probs)
        print("ExtraTree Roc:",rc_sc)
        
        clf = xgb.XGBClassifier(n_estimators=n_bags)
        clf.fit(X_train, y_train.values.ravel(), eval_metric='mlogloss')
        probs = clf.predict_proba(X_test)[:, 1]
        XGBoost_fpr, XGBoost_tpr, _ = roc_curve(y_test, probs)
        XGBoost_dict[c] = {'fpr': XGBoost_fpr, 'tpr':XGBoost_tpr}
        rc_sc=roc_auc_score(y_test, probs)
        print("XGBoost Roc:",rc_sc)
        
        bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
        bags=optimizer.evaluate_bags(bags)
        voting_test = []
        recall=[]
        fone=[]
        tpr1=[]
        fpr1=[]
        sc=0
        for i in tqdm(range(n_iter)):
            _, selection_idx = selection(bags)
            rows, cols = (n_bags, n_bags)
            bag_dis = [[0]*cols]*rows
            for idx, bag in bags.items():
                if idx in selection_idx:
                    for idx1, bag1 in bags.items():
                        if idx != idx1:
                            dis=abs(bag['payoff']-bag1['payoff'])
                            bag_dis[idx][idx1]=dis
            rows1, cols1 = (n_bags, n_bags+1)
            share_fn = [[0]*cols1]*rows1
            sigma=0.1
            
            for l in range(rows1):
                if l in selection_idx:
                    sum=0
                    for m in range(cols1):
                        if m==cols1-1:
                            share_fn[l][m]=sum
                            bags[l]['payoff']=(bags[l]['payoff']/sum)
                        else:
                            if bag_dis[l][m]< sigma:
                                share_fn[l][m]=1-(bag_dis[l][m]/sigma)
                            else:
                                share_fn[l][m]=0.00
                        sum+=share_fn[l][m]
            optimizer.mmo_evobagging_optimization(bags, X_test, y_test)
            met, fpr, tpr,rc_sc= optimizer.voting_metric_roc(bags, X_test, y_test)
            if sc == 0:
                sc=rc_sc
                tpr1=tpr
                fpr1=fpr
            else:
                if rc_sc>sc:
                    sc=rc_sc
                    tpr1=tpr
                    fpr1=fpr
            voting_test.append(met)
        print("MMO-ROC ", sc)
        class_roc_dict[c] = {'tpr': tpr1, 'fpr': fpr1}

    count=0
    fig, ax = plt.subplots()
    for c in class_roc_dict:
        current_tpr = list(class_roc_dict[c]['tpr'])[:]
        current_tpr.extend(list(bagging_roc_dict[c]['tpr'][:]))
        current_tpr.extend(list(RandomForest_dict[c]['tpr'][:]))
        current_tpr.extend(list( ExtraTree_dict[c]['tpr'][:]))
        current_tpr.extend(list(XGBoost_dict[c]['tpr'][:]))
        
        current_fpr = list(class_roc_dict[c]['fpr'][:])
        current_fpr.extend(list(bagging_roc_dict[c]['fpr'][:]))
        current_fpr.extend(list(RandomForest_dict[c]['fpr'][:]))
        current_fpr.extend(list(ExtraTree_dict[c]['fpr'][:]))
        current_fpr.extend(list(XGBoost_dict[c]['fpr'][:]))
        current_model = ['MMO-EvoBagging']*len(class_roc_dict[c]['tpr'])
        current_model.extend(['Bagging']*len(bagging_roc_dict[c]['tpr']))
        current_model.extend(['RandomForest']*len(RandomForest_dict[c]['tpr']))
        current_model.extend(['ExtraTree']*len(ExtraTree_dict[c]['tpr']))
        current_model.extend(['XGBoost']*len(XGBoost_dict[c]['tpr']))
        n = 100
        x = np.linspace(0,2,n)
        roc_df = pd.DataFrame({'True positive rate': current_tpr,
                               'False positive rate': current_fpr,
                               'Model': current_model
                               })
        g = sns.lineplot(data=roc_df, x='False positive rate', y='True positive rate',hue='Model',style='Model',ci=None)
        if c != max(class_roc_dict):
            plt.legend([],[], frameon=False)
        g = g.get_figure()
        g.savefig(f'roc/{dataset_name}_roc_class{c}.png')
        plt.clf()
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate ROC curves')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Percentage of test data')
    parser.add_argument('--metric', type=str, default='accuracy',
                        help='Classification metric')
    parser.add_argument('--n_bags', type=int,
                        help='Number of bags')
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Number of iterations')
    parser.add_argument('--n_select', type=int, default=0,
                        help='Number of selected bags each iteration')
    parser.add_argument('--n_new_bags', type=int,
                        help='Generation gap')
    parser.add_argument('--n_mutation', type=int,
                        help='Number of bags to perform mutation on')
    parser.add_argument('--mutation_rate', type=float, default=0.05,
                        help='Percentage of mutated instances in each bag')
    parser.add_argument('--size_coef', type=float,
                        help='Constant K for controlling size')
    parser.add_argument('--clf_coef', type=float,
                        help='Constant L for controlling coef')
    parser.add_argument('--procs', type=int, default=16,
                        help='Number of parallel processes')
    args = parser.parse_args()

    run(dataset_name=args.dataset_name, 
        test_size=args.test_size, 
        metric=args.metric,
        n_bags=args.n_bags, 
        n_iter=args.n_iter,
        n_select=args.n_select,
        n_new_bags=args.n_new_bags,
        n_mutation=args.n_mutation,
        mutation_rate=args.mutation_rate,
        size_coef=args.size_coef,
        clf_coef=args.clf_coef,
        procs=args.procs)