from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import sys
from datetime import datetime

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
# mine

from model import  *
from dataset import *
from utils import *
from solver import *
from iforest import iforest_test, iforest_test_epoch

import pdb

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='VAE + multi-classifier for out-of-distribution detection')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',help='learning rate ')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
    parser.add_argument('--use-cuda', default=True,help='use CUDA for training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', default=True,help='For Saving the current Model')

    parser.add_argument('--z-dim', type=int, default=20, help='autoencoder z dim: 20 for [mnist, cifar], 200 for [cifar100]')
    
    # dataset path
    
    parser.add_argument('--data-path', type=str, help='data path')
    parser.add_argument('--ano-data-path', type=str, help='anomaly data path')
    # result file path
    parser.add_argument('--result-path', type=str, help='test result path')
    parser.add_argument('--result-path-normal', type=str, help='test result path for normal dataset')
    parser.add_argument('--result-test-log-path', type=str, help='result-test-log-path')
    parser.add_argument('--result-test-log-path-normal', type=str, help='result-test-log-path for normal dataset')
    parser.add_argument('--result-train-loss-file', type=str, help='train result loss')
    parser.add_argument('--result-test-loss-file', type=str, help='test result loss')
    parser.add_argument('--result-test-loss-file-normal', type=str, help='test result loss for normal dataset')

    parser.add_argument('--result-measure-acc-true-file', type=str, help='result measure recon result: acc_true')
    parser.add_argument('--result-measure-prob-file', type=str, help='result measure prob result: fp tp fn tn acc auc prc')
    parser.add_argument('--result-measure-norm-file', type=str, help='result measure norm result: fp tp fn tn auc prc')
    parser.add_argument('--result-measure-recon-file', type=str, help='result measure recon result: fp tp fn tn auc prc')
    parser.add_argument('--save-model-path', type=str, help='model save path')


    
    parser.add_argument('--run-state', type=str, default='', help='train for train; test for test with all epochs\' models; test_single for the only one epoch test')
    # lambdas
    parser.add_argument('--dataset', type=str, default='', help='dataset [mnist, cifar, cifar100, svhn]')
    parser.add_argument('--anomaly-dataset', type=str ,default='', help='anomaly data type: [gaussian, uniform, svhn, lsun, imagenet, cifar10, cifar100]')

    parser.add_argument('--lambda-ae', type=float, default=0.0 , help='lambda for autoencode loss')
    parser.add_argument('--lambda-ce', type=float, default=0.0 , help='lambda for ce loss')
    parser.add_argument('--lambda-mem', type=float, default=0.0 , help='lambda for membership loss (in the membership loss)')
    parser.add_argument('--lambda-ce-sigmoid', type=float, default=0.0 , help='lambda for ce sigmoid loss')

    parser.add_argument('--unlabel-percent', type=float, default=0.0 , help='unlabel data\'s percent')
    parser.add_argument('--type-supervised', type=str, default='semi' , help='[semi,supervised],semi：半监督训练， supervised：有监督训练。默认半监督训练，当unlabel-percent=0.0，可认为有监督；')
    # 2020-02-01 change 
    # parser.add_argument('--lambda-ae-bool', type=bool, default=False , help='use ae loss or not')
    # parser.add_argument('--lambda-ce-bool', type=bool, default=False , help='use ce loss or not')
    # parser.add_argument('--lambda-mem-bool', type=bool, default=False , help='use mem loss or not')
    # parser.add_argument('--lambda-ce-sigmoid-bool', type=bool, default=False , help='use ce_sigmoid loss or not')
    
    parser.add_argument('--net-type', type=str, default='', help='mlp for mnist; [resnet18, resnet34] for cifar10 and cifar100')



    parser.add_argument('--lambda-norm', type=float, default=0.015 , help='lambda for norm loss')
    

    parser.add_argument('--type-name', type=str, default='prob', help='[prob, norm, recon], which model to load, only for test')
    
    parser.add_argument('--anomaly-data-number', type=int, default=10000, help='anomaly data numm numbers')
    parser.add_argument('--threshold-prob', type=float, default=0.99, help='max_prob threshold')
    parser.add_argument('--threshold-norm', type=float, default=0.26, help='norm threshold')
    parser.add_argument('--threshold-recon', type=float, default=0.6, help='autocoder recon  threshold')
    args = parser.parse_args()

    #pdb.set_trace()

    # init path file
    # if args.lambda_ae != 0.0:
    #     args.lambda_ae_bool = True
    # if args.lambda_ce != 0.0:
    #     args.lambda_ce_bool = True
    # if args.lambda_mem != 0.0:
    #     args.lambda_mem_bool = True
    # if args.lambda_ce_sigmoid != 0.0:
    #     args.lambda_ce_sigmoid_bool = True
    init_path_config(args)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda else "cpu")
    if args.run_state == 'train':
        if args.dataset not in ['mnist', 'cifar', 'cifar100', 'svhn']:
            print('main.py no dataset',args.dataset)
            return None
        multi_classifier, autoencoder, optimizer_multi_classifier, optimizer_autoencoder = init_model_for_train(args)
        train_loader, test_loader_normal = eval(args.dataset+'_dataset')(args).getdata(args)
        print('\nstart train all')
        for epoch in range(1, args.epochs+1):
            start_time = datetime.now()
            args.lr = lr_decay(epoch)
            print('\n'+'-'*50+'train'+'-'*150)
            print('epochs =',args.epochs,'\tepoch =',epoch, '\tdataset =',args.dataset, '\tae =',args.lambda_ae, '\tce =',args.lambda_ce, '\tmem =',args.lambda_mem,  '\tce_sigmoid =',args.lambda_ce_sigmoid, '\tnet_type =',args.net_type, '\tunlabel_percent =',args.unlabel_percent)
            
            train(args, multi_classifier, autoencoder, device, train_loader, optimizer_multi_classifier, optimizer_autoencoder, epoch)
            test_normal(args, multi_classifier, autoencoder, device, test_loader_normal, epoch)
            save_model(multi_classifier, autoencoder, args, epoch=epoch)
            
            end_time = datetime.now()
            print('train time cost :',end_time-start_time)
            if args.epochs == 1:
                with open('res_data/timming/'+args.dataset+'_timing.txt','a') as f:
                    content = args.net_type+'\t'
                    content += str(args.lambda_ae)+'\t'
                    content += str(args.lambda_ce)+'\t'
                    content += str(args.lambda_mem)+'\t'
                    content += str(args.lambda_ce_sigmoid)+'\t'
                    content += str(end_time-start_time)+'\n'
                    f.write(content)
        # train完 画loss的图
        loss_show(args)
    

    elif args.run_state == 'test_normal':
        if args.dataset not in ['mnist', 'cifar', 'cifar100', 'svhn']:
            print('main.py no dataset',args.dataset)
            return None
        
        #pdb.set_trace()
        train_loader, test_loader_normal = eval(args.dataset+'_dataset')(args).getdata(args)
        print('start test normal all')
        for epoch in range(1, args.epochs+1):
            start_time = datetime.now()
            autoencoder, multi_classifier = load_model(args, epoch)
            if autoencoder == None:
                print('the epoch', epoch, '\'s model doesn\'t exist!!! \nplease check the input!!!')
                return
            print('\n'+'-'*50+'test normal'+'-'*150)
            print('epochs =',args.epochs,'\tepoch =',epoch, '\tdataset =',args.dataset, '\tae =',args.lambda_ae, '\tce =',args.lambda_ce, '\tmem =',args.lambda_mem,  '\tce_sigmoid =',args.lambda_ce_sigmoid, '\tnet_type =',args.net_type, '\tunlabel_percent =',args.unlabel_percent)
            test_normal(args, multi_classifier, autoencoder, device, test_loader_normal, epoch)
            end_time = datetime.now()
            print('test time cost :',end_time-start_time)
    

    elif args.run_state == 'test':
        if args.anomaly_dataset in ['gaussian', 'uniform', 'cifar10', 'cifar100', 'svhn', 'lsun', 'imagenet']:
            #pdb.set_trace()
            test_laoder_anomaly = eval('ano_'+args.anomaly_dataset+'_dataset')(args).getdata(args)
        else:
            print('anomaly dataset: ',args.anomaly_dataset,'has not finished yet')
            return
        max_auc_prob = 0
        max_auc_norm = 0
        max_auc_recon = 0
        max_auc_iforest = 0
        #pdb.set_trace()
        print('start test all\n')
        for epoch in range(1, args.epochs+1):
            start_time = datetime.now()
            autoencoder, multi_classifier = load_model(args, epoch)
            if autoencoder == None:
                print('the epoch', epoch, '\'s model doesn\'t exist!!! \nplease check the input!!!')
                return
            print('\n'+'-'*50+'test'+'-'*150)
            print('epochs =',args.epochs,'\tepoch =',epoch, '\tdataset =',args.dataset, '\tdataset_ano =',args.anomaly_dataset, '\tae =',args.lambda_ae, '\tce =',args.lambda_ce, '\tmem =',args.lambda_mem,  '\tce_sigmoid =',args.lambda_ce_sigmoid, '\tnet_type =',args.net_type,  '\tunlabel_percent =',args.unlabel_percent)
            # prob
            auc_prob, prc_prob, tnr_at_tpr95_prob, auc_norm, prc_norm, tnr_at_tpr95_norm, auc_recon, prc_recon, tnr_at_tpr95_recon = test_anomaly(args, multi_classifier, autoencoder, device, test_laoder_anomaly, epoch)
            # iforest
            auc_iforest, prc_iforest, tnr_at_tpr95_iforest, prc_out_iforest = iforest_test_epoch(args.dataset, args.anomaly_dataset, args.net_type, args.lambda_ae, args.lambda_ce, args.lambda_mem, args.lambda_ce_sigmoid, args.unlabel_percent, epoch, args.type_supervised)
            # 存最优结果的
            max_auc_prob, max_auc_norm, max_auc_recon, max_auc_iforest = save_model_test_anomaly(args, epoch, auc_prob, prc_prob, tnr_at_tpr95_prob, max_auc_prob, auc_norm, prc_norm, tnr_at_tpr95_norm, max_auc_norm, auc_recon, prc_recon, tnr_at_tpr95_recon, max_auc_recon, auc_iforest, prc_iforest, tnr_at_tpr95_iforest, max_auc_iforest)
            end_time = datetime.now()
            print('test time cost :',end_time-start_time)
            
        auc_show(args)
        # # iforest
        # iforest_test(args.dataset, args.anomaly_dataset, args.net_type, args.lambda_ae, args.lambda_ce, args.lambda_mem, args.lambda_ce_sigmoid)
    elif args.run_state == 'iforest':
        iforest_test(args.dataset, args.anomaly_dataset, args.net_type, args.lambda_ae, args.lambda_ce, args.lambda_mem, args.lambda_ce_sigmoid, args.unlabel_percent, args.type_supervised)

if __name__ == '__main__':
    main()
