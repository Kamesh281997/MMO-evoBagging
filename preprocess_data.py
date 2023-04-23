from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder

def spiral_xy(i, spiral_num):
    """
    Create the data for a spiral.

    Arguments:
        i runs from 0 to 96
        spiral_num is 1 or -1
    """
    φ = i/16 * math.pi
    r = 6.5 * ((104 - i)/104)
    x = (r * math.cos(φ) * spiral_num)/13 + 0.5
    y = (r * math.sin(φ) * spiral_num)/13 + 0.5
    return (x, y)

def spiral(spiral_num):
    return [spiral_xy(i, spiral_num) for i in range(97)]

class Object(object):
    pass

def load_data_sklearn(dataset_name):
    if dataset_name == 'mnist':
        data = load_digits()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    return data

def load_abalone(data):
    df = pd.read_csv('data/abalone.csv')
    df['Sex'] = df['Sex'].apply(lambda x: 0 if x=='M' else 1)
    labels = []
    for r in df['Rings']:
        if 0<=r and r<=7:
            label = 1
        elif 8<=r and r<=10:
            label = 2
        elif 11<=r and r<=15:
            label = 3
        elif r>15:
            label = 4
        labels.append(label)
    data.data = np.asarray(df.loc[:, df.columns!='Rings'])
    data.target = np.asarray(labels)

    return data

def load_red_wine(data):
    df = pd.read_csv('data/winequality-red.csv')
    df.quality = np.where(df.quality < 6.5, 0, 1)
    data.data = np.asarray(df.loc[:, df.columns!='quality'])
    data.target = np.asarray(df['quality'])

    return data

def load_pima(data):
    df = pd.read_csv('data/diabetes.csv')
    data.data = np.asarray(df.loc[:, df.columns!='Outcome'])
    data.target = np.asarray(df['Outcome'])
    return data

def load_car(data):
    df = pd.read_csv('data/car.data', sep=',', header=None)
    mapping = {'acc': 1, 'good': 2, 'unacc': 0, 'vgood': 3}
    df.iloc[:, -1] = df.iloc[:, -1].map(mapping)
    enc = OneHotEncoder()
    enc.fit(np.asarray(df.iloc[:, :-1]))
    data.data = np.asarray(enc.transform(np.asarray(df.iloc[:, :-1])).toarray())
    data.target = np.asarray(df.iloc[:, -1])
    return data

def load_tictactoe(data):
    df = pd.read_csv('data/tic-tac-toe.data', sep=',', header=None)
    mapping = {'negative': 0, 'positive': 1}
    df.iloc[:, -1] = df.iloc[:, -1].map(mapping)
    enc = OneHotEncoder()
    enc.fit(np.asarray(df.iloc[:, :-1]))
    data.data = np.asarray(enc.transform(np.asarray(df.iloc[:, :-1])).toarray())
    data.target = np.asarray(df.iloc[:, -1])
    return data

def load_ionosphere(data):
    df = pd.read_csv('data/ionosphere.data', sep=',', header=None)
    mapping = {'b': 0, 'g': 1}
    df.iloc[:, -1] = df.iloc[:, -1].map(mapping)
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(df.iloc[:, -1])
    return data

def load_churn(data):
    df = pd.read_csv('data/churn.tsv', sep='\t')
    data.data = np.asarray(df.loc[:, df.columns!='target'])
    data.target = np.asarray(df['target'])
    return data

def load_flare(data):
    df = pd.read_csv('data/flare.tsv', sep='\t')
    data.data = np.asarray(df.loc[:, df.columns!='target'])
    data.target = np.asarray(df['target'])
    # smt = BorderlineSMOTE()
    # data.data, data.target = smt.fit_resample(data.data, data.target)
    return data

def load_ring(data):
    df = pd.read_csv('data/ring.tsv', sep='\t')
    data.data = np.asarray(df.loc[:, df.columns!='target'])
    data.target = np.asarray(df['target'])
    return data    

def load_nbit(data, n_bit):
    data_dict = dict()
    while len(data_dict) < 2**n_bit:
        new_seq = np.random.randint(0, 2, n_bit)
        s = ''.join([str(bit) for bit in new_seq])
        if s not in data_dict.keys():
            if np.unique(new_seq, return_counts=True)[1][0]%2 == 0:
                data_dict[s] = 1
            else:
                data_dict[s] = 0
    data.data = []
    data.target = []
    for bits, label in data_dict.items():
        bits = [int(bit) for bit in bits]
        data.data.append(bits)
        data.target.append(label)
    data.data = np.asarray(data.data)
    data.target = np.asarray(data.target)
    return data

def load_two_spiral(data):
    a = spiral(1)
    data.data = a[:]
    data.data.extend(spiral(-1))
    data.data = np.asarray(data.data)
    data.target = [0]*len(a)
    data.target.extend([1]*len(a))
    data.target = np.asarray(data.target)
    return data

def load_statlog(data):
    df = pd.read_csv('data/statlog.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(df.iloc[:, -1])
    data.data = np.asarray(df.iloc[:, :-1])
    
    data.target = np.asarray(y_train)
    return data 

def load_kidney(data):
    df = pd.read_csv('data/kidney_disease.csv')
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(df.iloc[:, -1])
    return data

def load_heart_cleveland(data):
    df = pd.read_csv('data/heart_cleveland.csv')
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(df.iloc[:, -1])
    return data

def load_dermatalogy(data):
    df = pd.read_csv('data/dermatalogy.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(df.iloc[:, -1])
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(y_train)
    return data

def  load_thyroid(data):
    df = pd.read_csv('data/thyroid.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(df.iloc[:, -1])
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(y_train)
    return data

def  load_parkinson(data):
    df = pd.read_csv('data/parkinson1.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(df.iloc[:, -1])
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(y_train)
    # oversample = BorderlineSMOTE()
    # data.data, data.target  = oversample.fit_resample(data.data , data.target )
    return data

def  load_heart_attack(data):
    df = pd.read_csv('data/heart_attack1.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(df.iloc[:, -1])
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(y_train)
    return data

def  load_heart_failure(data):
    df = pd.read_csv('data/heart_failure.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(df.iloc[:, -1])
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(y_train)
    return data

def  load_hepatitis(data):
    df = pd.read_csv('data/hepatitis.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(df.iloc[:, -1])
    data.data = np.asarray(df.iloc[:, :-1])
    data.target = np.asarray(y_train)
    return data

def load_data(dataset_name, test_size=0.1, random_state=1):
    sc = StandardScaler()
    data = Object()
    if dataset_name in ["mnist", "breast_cancer"]:
        data = load_data_sklearn(dataset_name)
    elif dataset_name == 'abalone':
        data = load_abalone(data)
    elif dataset_name == 'red_wine':
        data = load_red_wine(data)
    elif dataset_name == 'pima':
        data = load_pima(data)
    elif dataset_name == 'car':
        data = load_car(data)
    elif dataset_name == 'tic-tac-toe':
        data = load_tictactoe(data)
    elif dataset_name == 'ionosphere':
        data = load_ionosphere(data)
    elif dataset_name == 'churn':
        data = load_churn(data)
    elif dataset_name == 'flare':
        data = load_flare(data)
    elif dataset_name == 'ring':
        data = load_ring(data)
    elif dataset_name == '4bit':
        data = load_nbit(data, 4)
    elif dataset_name == '6bit':
        data = load_nbit(data, 6)
    elif dataset_name == '8bit':
        data = load_nbit(data, 8)
    elif dataset_name == 'two-spiral':
        data = load_two_spiral(data)
    elif dataset_name == 'statlog':
        data = load_statlog(data)
    elif dataset_name == 'kidney_disease':
        data = load_kidney(data)
    elif dataset_name == 'heart_cleveland':
        data = load_heart_cleveland(data)
    elif dataset_name == 'dermatalogy':
        data = load_dermatalogy(data)
    elif dataset_name == 'thyroid':
        data = load_thyroid(data)
    elif dataset_name == 'parkinson':
        data = load_parkinson(data)
    elif dataset_name == 'heart_attack':
        data = load_heart_attack(data)
    elif dataset_name == 'heart_failure':
        data = load_heart_failure(data)    
    elif dataset_name == 'hepatitis':
        data = load_hepatitis(data)  
    X, y = data.data, data.target
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=test_size, 
                                                            random_state=random_state
                                                            )
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # if dataset_name == "abalone":
        #     principal=PCA(n_components=5)
        #     principal.fit(X_train )
        #     principal.fit(X_test )
        #     X_train=principal.transform(X_train)
        #     X_test=principal.transform(X_test)
        #     print("X_train ",X_train.shape)
        #     print("X_test ",X_test.shape)
 
# Check the dimensions of data after PCA

        print('Train: ', X_train.shape, ' | Test: ', X_test.shape)
        print('Train labels: ', np.unique(y_train, return_counts=True))
        print('Test labels: ', np.unique(y_test, return_counts=True))
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        return X_train, X_test, y_train, y_test
    else:
        X_train = sc.fit_transform(X)
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y)
        return X_train, None, y_train, None