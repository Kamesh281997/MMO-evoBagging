from preprocess_data import load_data
from mmo_evoBagging import *
import seaborn as sns
import yaml
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")
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
            selected_bag_dict = {i: bags[i] for i in selected_ids}
            return None, selected_ids
     

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
    if dataset_name !="two-spiral":
        oversample = BorderlineSMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    optimizer = MMO_EvoBagging(X_train, y_train,  X_test, y_test,  n_bags,n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef,clf_coef, metric, procs)
    avg_fitness = []
    # init random bags of random sizes
    bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
    # evaluate
    bags = optimizer.evaluate_bags(bags)
    # bags=optimizer.re_evaluate_bags(bags, X_test, y_test)
    avg_fitness.append(np.mean([bag["payoff"] for bag in bags.values()]))
    num = 9
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
        avg_fitness.append(np.mean([bag["payoff"] for bag in bags.values()]))
    return avg_fitness

def run(dataset_name, hyp, l_hyp):
    with open("configs.yml", "r") as fh:
        configs = yaml.load(fh, Loader=yaml.SafeLoader)
    dataset_config = configs[dataset_name]
    plot_df = pd.DataFrame(columns=["Value", "Iter", "Average Fitness"])
    count=0
    fig, ax = plt.subplots()
    for value in l_hyp:
        if hyp != "mutation_rate":
            dataset_config[hyp] = value
            mutation_rate = dataset_config['mutation_rate']
        else:
            mutation_rate = value
        avg_fitness = run_dataset(dataset_name,
                                metric='accuracy',
                                n_bags=dataset_config['n_bags'], 
                                n_iter=dataset_config['n_iter'],
                                n_select=0,
                                n_new_bags=dataset_config['n_new_bags'],
                                n_mutation=dataset_config['n_mutation'],
                                mutation_rate=mutation_rate,
                                size_coef=dataset_config['size_coef'], 
                                clf_coef=dataset_config['clf_coef'], 
                                procs=4)
        val=str(value)
        val=val+" (M)"
        if count==0:
            ax.plot(range(dataset_config['n_iter'] + 1),avg_fitness,label=val,linestyle="solid")
        elif count==1:
            ax.plot(range(dataset_config['n_iter'] + 1),avg_fitness,label=val,linestyle="-.")
        elif count==2:
            ax.plot(range(dataset_config['n_iter'] + 1),avg_fitness,label=val,linestyle=":")
        elif count==3:
            ax.plot(range(dataset_config['n_iter'] + 1),avg_fitness,label=val,linestyle="--")
        count=count+1
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Fitness")
    ax.legend()
    fig = ax.get_figure()
    fig.savefig(f"images/hyp_{hyp}_{dataset_name}.png")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment reducing bias')
    parser.add_argument('--hyp', type=str, default="mutation_rate",
                        help="Hyperparams to experiment")
    parser.add_argument('--l_hyp', type=int, default=[0.01, 0.05, 0.08, 0.1, 0.2],
                        # [0.01, 0.05, 0.08, 0.1, 0.2]
                        help='<Required> List of values for hyperparam')

    parser.add_argument('--dataset_name', type=str, default="pima",
                        help="Dataset name")
    args = parser.parse_args()
    run(args.dataset_name, args.hyp, args.l_hyp)