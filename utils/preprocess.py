from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from make_dataset.Datasets import Datasets
import pandas as pd
from sklearn import metrics
def construct_data(data, feature_map, labels=0):
    res = []
    # get features
    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # append labels as last
    sample_n = len(res[0])
    # get label
    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)
    return res

def eva(y_true, y_pred, epoch=0):
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    pre = metrics.precision_score(y_true, y_pred, average='macro')
    rec = metrics.recall_score(y_true, y_pred, average='macro')

    return acc, f1, pre, rec

def find_indices(arr):
    index_dict = {}
    for index, value in enumerate(arr):
        if value not in index_dict:
            index_dict[value] = [index]
        else:
            index_dict[value].append(index)
    return index_dict


# get feature list
def get_feature_map(feature_list_path):
    feature_file = open(feature_list_path, 'r')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    return feature_list

def add_additional_tokens(model_path):
    token_list = ['{:016b}'.format(1 << i) for i in range(16)]
    all_token_list = []
    for i in range(120):
        for j in token_list:
            token = str(i) + j
            all_token_list.append(token)
    print("all_token_list", len(all_token_list))

    lm_model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': all_token_list})
    tokenizer.save_pretrained(model_path)

def get_dataset(train_path,  feature_list_path):
    train = pd.read_csv(train_path, sep=',')
    # train = train.iloc[0:5000, :]
    print("train:{}".format(train.shape))
    # get features
    feature_map = get_feature_map(feature_list_path)
    Fea_num = len(feature_map)
    # train_dataset_indata = construct_data(train, feature_map, labels=train.attack.tolist())
    # train_dataset_indata = construct_data(train, feature_map, labels=train.attack.tolist())
    train_dataset_indata = construct_data(train, feature_map, labels=train.label.tolist())
    train_dataset = Datasets(train_dataset_indata)

    return feature_map, train_dataset, Fea_num