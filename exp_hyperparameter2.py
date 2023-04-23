from preprocess_data import load_data
from mmo_evoBagging import *
import seaborn as sns
import yaml
import argparse
from imblearn.over_sampling import BorderlineSMOTE
import warnings
warnings.filterwarnings("ignore")

def run_dataset(dataset_name, 
                metric,
                n_bags, 
                n_iter,
                n_select,
                n_new_bags,
                n_mutation,
                mutation_rate,
                size_coef, 
                clf_coef,
                procs=4):
    X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=0.2)
    oversample = BorderlineSMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    optimizer = MMO_EvoBagging(X_train, y_train,  X_test, y_test,  n_bags,n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef,clf_coef, metric, procs)
    avg_size, avg_depth = 0, 0
    # init random bags of random sizes
    bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
    # evaluate
    bags = optimizer.evaluate_bags(bags)
    for i in range(n_iter):
        rows, cols = (n_bags, n_bags)
        bag_dis = [[0]*cols]*rows
        for idx, bag in bags.items():
            for idx1, bag1 in bags.items():
                if idx != idx1:
                    dis=abs(bag['payoff']-bag1['payoff'])
                    bag_dis[idx][idx1]=dis
        rows1, cols1 = (n_bags, n_bags+1)
        share_fn = [[0]*cols1]*rows1
        sigma=50
        for l in range(rows1):
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
    avg_size = np.mean([bag["size"] for bag in bags.values()])
    met = optimizer.voting_metric(X_test, y_test, bags, False)   
    return avg_size, met

def run(dataset_name, hyp, l_hyp):
    with open("configs.yml", "r") as fh:
        configs = yaml.load(fh, Loader=yaml.SafeLoader)
    dataset_config = configs[dataset_name]
    for value in l_hyp:
        if hyp != "size_coef":
            dataset_config[hyp]=value
            size_value=dataset_config['size_coef']
        else:
            size_value=value
        avg_size, met = run_dataset(dataset_name,
                                metric='accuracy',
                                n_bags=dataset_config['n_bags'], 
                                n_iter=dataset_config['n_iter'],
                                n_select=0,
                                n_new_bags=dataset_config['n_new_bags'],
                                n_mutation=dataset_config['n_mutation'],
                                mutation_rate=dataset_config['mutation_rate'],
                                size_coef=size_value, 
                                clf_coef=dataset_config['clf_coef'], 
                                procs=4)
        print(value, avg_size, met)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment reducing bias')
    parser.add_argument('--hyp', type=str, default="size_coef",
                        help="Hyperparams to experiment")
    parser.add_argument('--l_hyp', type=int, default=[1000, 3000, 5000, 10000, 15000], 
                        help='<Required> List of values for hyperparam')
    parser.add_argument('--dataset_name', type=str, default="pima",
                        help="Dataset name")
    args = parser.parse_args()
    run(args.dataset_name, args.hyp, args.l_hyp)