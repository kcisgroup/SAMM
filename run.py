# -*- coding: utf-8 -*-
import time
import os
import torch.nn as nn
import numpy as np
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.preprocess import *
from model.SAMM import SAMM
from datetime import datetime
import argparse
from make_dataset.Datasets import Datasets
from torch.optim import lr_scheduler
import warnings
import logging

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.ERROR)
# 忽略特定警告类
warnings.filterwarnings("ignore", message="You are resizing the embedding layer without providing a pad_to_multiple_of parameter.*")
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def train_epoch(model, epoch, train_loader, optimizer):
    t = time.time()
    total_loss = []
    all_labels = []
    all_clusters = []
    model.train()
    flag = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = [item.float().to(model.device) for item in [data, label]]
        optimizer.zero_grad()
        if flag == 0 and epoch == 0:
            print('data shape:', data.shape)
        sen_ids, all_loss, clusters = model(data)
        # all_loss, clusters = model(data)

        loss = all_loss
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        flag = 1



    print('Epoch: {:03d}'.format(epoch + 1),
          'total_loss: {:.4f}'.format(np.mean(total_loss)),
          'time: {:.4f}s'.format(time.time() - t))

    # print('\nACC: {:.4f} | f1: {:.4f} | PRE:{:.4f} | REC : {:.4f}\n'.format(acc, f1, pre, rec))

def test_epoch(model, data_loader, e):
    model.eval()
    all_labels = []
    all_clusters = []
    for x, label in data_loader:
        x, label = [item.float().to(model.device) for item in [x, label]]
        with torch.no_grad():
            sen_ids, all_loss, clusters = model(x)
            # all_loss, clusters = model(x)
        if len(all_clusters) <= 0:
            all_clusters = clusters
            all_labels = label
        else:
            all_clusters = torch.cat((all_clusters, clusters), dim=0)
            all_labels = torch.cat((all_labels, label), dim=0)

    print('all_clusters:', all_clusters.shape)
    print('all_labels:', all_labels.shape)

    all_labels = all_labels.cpu().detach().numpy()
    all_clusters = all_clusters.cpu().detach().numpy()
    unique_values, counts = np.unique(all_clusters, return_counts=True)
    print("final clusters:")
    for i in range(len(unique_values)):
        print(unique_values[i], counts[i])

    cluster_indices = find_indices(all_clusters.reshape((all_clusters.shape[0])))
    predicted_labels = np.zeros_like(all_labels)
    for key, value in cluster_indices.items():
        print(f"clusters: {key}")
        labels = all_labels[value]
        unique_values, counts = np.unique(labels, return_counts=True)
        for i in range(len(unique_values)):
            print(unique_values[i], counts[i])
        counter = Counter(labels)
        most_common_element, count = counter.most_common(1)[0]
        # print(f'The element that appears most often is {most_common_element}, which appears {count} times.')
        predicted_labels[value] = most_common_element

    f1, purity = eva(all_labels, predicted_labels)
    return f1, purity, all_labels, all_clusters




def solver(args, model, train_loader):
    purity_list = []
    f1_list = []
    best_labels = []
    best_clusters = []
    best_f1 = 0
    best_epoch = 0
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.wd)
    print("learning_rate:{}".format(args.lr))
    for e in range(args.epoch):
        train_epoch(model, e, train_loader, optimizer)
        infer_start = time.perf_counter()
        f1, purity, all_labels, all_clusters = test_epoch(model, train_loader, e)
        infer_end = time.perf_counter()
        infer_time = infer_end - infer_start
        print(f"inference time:{infer_time:.6f}")
        if best_f1 < f1:
            best_f1 = f1
            best_epoch = e+1
            best_labels = all_labels
            best_clusters = all_clusters

        purity_list.append(purity)
        f1_list.append(f1)

        print('\nEpoch: {:02d} | Purity: {:.4f} | f1: {:.4f} \n'.format(e + 1, purity, f1))

    print("best epoch:", best_epoch)
    unique_values, counts = np.unique(best_clusters, return_counts=True)
    print("final clusters:")
    for i in range(len(unique_values)):
        print(unique_values[i], counts[i])

    cluster_indices = find_indices(best_clusters.reshape((all_clusters.shape[0])))
    predicted_labels = np.zeros_like(best_labels)
    for key, value in cluster_indices.items():
        print(f"clusters: {key}")
        labels = best_labels[value]
        unique_values, counts = np.unique(labels, return_counts=True)
        for i in range(len(unique_values)):
            print(unique_values[i], counts[i])
        counter = Counter(labels)
        most_common_element, count = counter.most_common(1)[0]
        # print(f'The element that appears most often is {most_common_element}, which appears {count} times.')
        predicted_labels[value] = most_common_element

    f1, purity = eva(best_labels, predicted_labels)


    print('\nPurity: {:.4f} | f1: {:.4f} \n'.format(purity, f1))

    return purity_list, f1_list


if __name__ == '__main__':
    starttime = datetime.now()
    parser = argparse.ArgumentParser(description='Adaptive clustering algorithm based on symbolization and large language model')

    # Dataset parameters
    parser.add_argument('--train_path', default='../datas/breast_cancer_wisconsin_original/test.csv', help='dataset path')
    parser.add_argument('--feature_list', default='../datas/breast_cancer_wisconsin_original/list.txt', help='feature list path')
    parser.add_argument('--feature_dim', default=9, type=int, help='the feature num of data point')
    parser.add_argument('--batch_size', type=int, default=128, help='the number for a batch')

    # Large Language model parameters
    parser.add_argument('--lm_path', default='./LM/BertForMaskedLM', help='Large Language Model Path')
    parser.add_argument('--token_flag', default=True, type=bool, help='whether to add additional tokens')
    parser.add_argument('--pretrain', type=bool, default=False, help='whether use pretrained model')

    # model parameters
    parser.add_argument('--epoch', default=100, type=int, help='the number of train epochs')
    parser.add_argument('--cluster_space', default=32, type=int, help='the space for clustering')
    parser.add_argument('--word_emb_dim', default=32, type=int, help='the dimension of word embedding')
    parser.add_argument('--n_res_layers', default=1, type=int, help='the time of residual layers')
    parser.add_argument('--hidden_dim', default=32, type=int, help='the dimension of hidden layer')
    parser.add_argument('--symbol_space', default=16, type=int, help='the symbol space for a feature')
    parser.add_argument('--fea_emb_dim', default=1024, type=int, help='the dimension of the first linear')
    parser.add_argument('--lr', default=1e-3, type=float, help='the learning rate of the model')
    parser.add_argument('--wd', default=0, type=float, help='the decay of weight')

    # device parameters
    parser.add_argument('--gpu_id', default=0, type=int, help='which gpu of the device')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use GPU')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging the train status')

    args = parser.parse_args()

    # add tokens
    if args.token_flag:
        add_additional_tokens(args.lm_path)

    # Load data
    feature_map, train_dataset, Fea_num = get_dataset(args.train_path, args.feature_list)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    # Main body
    model = SAMM(args)
    model.to(model.device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    purity_list, f1_list = solver(args, model, train_loader)
    endtime = datetime.now()
    t = (endtime - starttime).seconds
    print('*************************The total time is ', t)



