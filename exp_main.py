from preprocess_data import load_data
from mmo_evoBagging import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.decomposition import PCA
import seaborn as sns
import argparse
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
def selection( bags, mode="selection"):
        selected_bag_dict = {}
        selected_ids = []
        bag_idx, payoff_list = [], []
        for idx, bag in bags.items():
            bag_idx.append(idx)
            payoff_list.append(bag['payoff'])
        if mode=="selection":
            selected_ids = [idx for _, idx in sorted(zip(payoff_list, bag_idx), reverse=True)][:9]
            selected_bag_dict = {i: bags[i] for i in selected_ids}
            return None, selected_ids
def run(dataset_name, 
        test_size, 
        n_exp,
        metric,
        n_bags, 
        n_iter,
        n_select,
        n_new_bags,
        n_mutation,
        mutation_rate,
        size_coef, 
        clf_coef,
        voting='majority',
        procs=4):
    
    X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=test_size)
    oversample = BorderlineSMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    optimizer = MMO_EvoBagging(X_train, y_train,  X_test, y_test,  n_bags,n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef,clf_coef, metric, procs)
    all_voting_train, all_voting_test = [], []
    depth_evobagging = []
    for t in tqdm(range(n_exp)):
        bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
        payoff_df = pd.DataFrame(columns=['bag' + str(i) for i in range(n_bags)])
        size_df = pd.DataFrame(columns=['bag' + str(i) for i in range(n_bags)])
        depth_df = pd.DataFrame(columns=['bag' + str(i) for i in range(n_bags)])  
        bags = optimizer.evaluate_bags(bags)
        voting_train, voting_test = [], []
        weighted_voting_test = []
        for i in range(n_iter):
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
            bags = optimizer.mmo_evobagging_optimization(bags, X_test, y_test)
            payoff_df.loc[i + 1, :] = [round(bags[j]['payoff']*100, 1) for j in range(n_bags)]
            size_df.loc[i + 1, :] = [round(bags[j]['size'], 1) for j in range(n_bags)]
            majority_voting_train_metric = optimizer.voting_metric(X_train, y_train, bags)
            majority_voting_test_metric = optimizer.voting_metric(X_test, y_test, bags)
            voting_train.append(round(majority_voting_train_metric, 2))
            voting_test.append(round(majority_voting_test_metric, 2))
            print("Train:",voting_train)
            print("Test:",voting_test)
            if voting == 'weighted':
                weighted_voting_test_metric = optimizer.voting_metric_weighted(X_test, y_test, bags)*100
                weighted_voting_test.append(weighted_voting_test_metric)
        print_df = payoff_df.mean(axis=1)
        p = sns.lineplot(data=print_df)
        p.set_xlabel("iteration")
        p.set_ylabel("fitness")
        p.xaxis.set_major_locator(ticker.MultipleLocator(5))
        p.xaxis.set_major_formatter(ticker.ScalarFormatter())
        fig = p.get_figure()
        fig.savefig(f'images/fitness/{dataset_name}_{t}.png')
        plt.clf()
        if voting == 'weighted':
            voting_print_df = pd.DataFrame({'Majority': voting_test,'Weighted': weighted_voting_test})
            p = sns.lineplot(data=voting_print_df)
            p.set_xlabel("iteration")
            p.set_ylabel("accuracy")
            p.xaxis.set_major_locator(ticker.MultipleLocator(5))
            p.xaxis.set_major_formatter(ticker.ScalarFormatter())
            fig = p.get_figure()
            fig.savefig(f'images/voting/{dataset_name}_{t}.png')
            plt.clf()
        best_iter = np.argmax(voting_train)
        all_voting_train.append(voting_train[best_iter])
        all_voting_test.append(voting_test[best_iter])
        print(all_voting_test)
        print("Training accuracy")
        print(np.mean(all_voting_train))
        print("Test accuracy")
        print(np.mean(all_voting_test))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Main experiment for real datasets')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Percentage of test data')
    parser.add_argument('--n_exp', type=int, default=30,
                        help='Number of experiments')
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
                        help='Constant P for controlling Bag Performance')
    parser.add_argument('--voting', type=str, default='majority',
                        help='Type of voting rule')
    parser.add_argument('--procs', type=int, default=16,
                        help='Number of parallel processes')
    args = parser.parse_args()

    run(dataset_name=args.dataset_name, 
        test_size=args.test_size, 
        n_exp=args.n_exp,
        metric=args.metric,
        n_bags=args.n_bags, 
        n_iter=args.n_iter,
        n_select=args.n_select,
        n_new_bags=args.n_new_bags,
        n_mutation=args.n_mutation,
        mutation_rate=args.mutation_rate,
        size_coef=args.size_coef,
        clf_coef=args.clf_coef,
        voting=args.voting,
        procs=args.procs)
