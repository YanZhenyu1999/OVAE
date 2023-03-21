import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import argparse
import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import time

from dataset import *
from model import *

import pdb

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def lr_decay(epoch):
    if epoch < 20:
        return 0.1
    if epoch < 150:
        return 0.01
    else:
        return 0.001

# membership loss 现在不用
def membership_loss(output, target):
    lambda_mem = 1
    target = target.view(-1,1)
    target_prob = torch.gather(input=output,dim=1,index=target)
    k = output.shape[1]
    batch_size = output.shape[0]
    # lambda - 2*lambda*c_i + (lambda-1/(k-1))*c_i^2 + (1/(k-1))*\sum_(all) c_j^2
    loss = -2*lambda_mem*target_prob
    loss += (lambda_mem-1/(k-1)) * (target_prob.mul(target_prob))
    loss += output.mul(output).sum(1).div(k-1).view(-1,1)
    loss += lambda_mem
    loss = loss.sum().div(batch_size)
    return loss

# sigmoid分类的loss
def ce_sigmoid_loss(output, target):
    loss_func = nn.BCELoss()
    loss = loss_func(output, target)
    return loss

# x的重构误差计算
def recon_loss(x_hat, data, batch_size):
    loss = F.mse_loss(x_hat, data, reduction='sum').div(batch_size)
    return loss


# for vae KL散度计算
def kl_divergence(mu, logvar):
    # print('mu: ',mu.size())
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld

# for vae 重参数方法
def reparameterizetion(z, z_dim):
        mu= z[:,:z_dim]
        logvar = z[:,z_dim:]
        pdb.set_trace()
        
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return mu, logvar, eps.mul(std).add_(mu)

# 计算混淆矩阵结果，为计算auc、prc用
def confusion_matrix(target_list, pred_list):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(target_list)):
        if target_list[i] == 1 and pred_list[i] == 1:
            tp += 1
        elif target_list[i] == 1 and pred_list[i] == 0:
            fn += 1
        elif target_list[i] == 0 and pred_list[i] == 1:
            fp += 1
        elif target_list[i] == 0 and pred_list[i] == 0:
            tn += 1
    return tp, fn, fp, tn


# 计算tnr_at_tpr95 现在没用
def tnr_at_tpr95_cal(ano_target_list, prob_list):
    positive_prod_list = prob_list[:10000]
    positive_prod_list.sort()
    threshold = positive_prod_list[500]
    pred_list = []
    for i in range(len(prob_list)):
        if prob_list[i] >= threshold:
            pred_list.append(1)
        else:
            pred_list.append(0)
    true_negative = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(ano_target_list, pred_list))
    actual_negative = len(ano_target_list) - sum(ano_target_list)
    tnr = true_negative / actual_negative
    return tnr

# 移动所有的test log数据到对应的异常数据集下 
def load_normal_test_data(args, epoch):
    # cp normal_test_log/epoch_99* anomaly_test_log/
    filename =  args.result_test_log_path+'epoch_'+str(int(epoch))+'.txt'
    cmd = 'cp ' + str(args.result_test_log_path_normal) + 'epoch_'+str(int(epoch))+'.txt '+ str(args.result_test_log_path)
    subprocess.call(cmd, shell=True)
    cmd = 'cp ' + str(args.result_test_log_path_normal) + 'epoch_'+str(int(epoch))+'_*.txt '+ str(args.result_test_log_path)
    subprocess.call(cmd, shell=True)
    df = pd.read_csv(filename, '\t')
    # 分别为2 3 4 5 6 7 8
    ano_target_list = list(df['ano_target']) #异常检测里面的类别标签
    prod_list = list(df['max_prob']) #最大概率值
    ano_pred_prob_list = list(df['ano_pred_prob']) #异常检测里面的预测类别 prob
    norm_list = list(df['norm']) #真实的norm结果
    ano_pred_norm_list = list(df['ano_pred_norm']) #异常检测里面的预测类别 norm
    recon_list = list(df['recon']) #真实的recon结果
    ano_pred_recon_list = list(df['ano_pred_recon']) #异常检测里面的预测类别 recon
    return ano_target_list, prod_list, ano_pred_prob_list, norm_list, ano_pred_norm_list, recon_list, ano_pred_recon_list



# 路径初始化，用于init
def lambda_for_path(args):
    path_result_source = '/data/lq/result/anomaly_detection_result/out_of_distribution/'+str(args.dataset)+'/'

    path_name_lambda = 'net_type_'+str(args.net_type)+'/'
    path_name_lambda += 'lambda_ae_'+str(args.lambda_ae)+'/'
    path_name_lambda += 'lambda_ce_'+str(args.lambda_ce)+'/'
    path_name_lambda += 'lambda_mem_'+str(args.lambda_mem)+'/'
    path_name_lambda += 'lambda_ce_sigmoid_'+str(args.lambda_ce_sigmoid)+'/'
    if args.type_supervised == 'semi':
        path_name_lambda += 'unlabel_percent_'+str(args.unlabel_percent)+'/'
    elif args.type_supervised == 'supervised':
        path_name_lambda += 'supervised_without_unlabel_percent_'+str(args.unlabel_percent)+'/'

    path_lambda = ''
    path_lambda += str(args.dataset)+'_'
    path_lambda += str(args.anomaly_dataset)+'_'
    path_lambda += str(args.net_type)+'_'
    path_lambda += str(float(args.lambda_ae))+'_'
    path_lambda += str(float(args.lambda_ce))+'_'
    path_lambda += str(float(args.lambda_mem))+'_'
    path_lambda += str(float(args.lambda_ce_sigmoid))+'_'
    if args.type_supervised == 'semi':
        path_lambda += str(float(args.unlabel_percent))
    elif args.type_supervised == 'supervised':
        path_lambda += 'supervised_'+str(float(args.unlabel_percent))

    path_result_train = path_result_source + 'train/' + path_name_lambda
    path_result_test = path_result_source + 'test/' + 'anomaly_dataset_' + str(args.anomaly_dataset) + '/' + path_name_lambda

    return path_result_source, path_result_train, path_result_test, path_lambda, path_name_lambda


# 初始化main的各种参数，根据各种输入的超参数，初始化
def init_path_config(args):
    # normaly dataset
    if args.dataset not in ['mnist', 'cifar', 'cifar100', 'svhn']:
        print('no the dataset: ',args.dataset)
        return
    if args.dataset == 'mnist' and args.net_type != 'mlp':
        print('only mlp is accepted for mnist, the',args.net_type, 'is not right')
    if args.dataset in ['cifar','cifar100','svhn'] and args.net_type not in ['resnet18', 'resnet34', 'densenet121', 'densenet169', 'densenet201']:
        print(args.net_type, 'is not for', args.dataset)
        return
    # net_type
    if args.net_type not in ['mlp', 'resnet18', 'resnet34', 'densenet121', 'densenet169', 'densenet201']:
        print('no the net',args.net_type)
        return


    if args.anomaly_dataset in ['gaussian','uniform']:
        args.ano_data_path = ''
    elif args.anomaly_dataset in ['svhn', 'lsun', 'imagenet', 'cifar100']:
        args.ano_data_path = '/data/lq/dataset/'+str(args.anomaly_dataset)+'/'
    elif args.anomaly_dataset == 'cifar10':
        args.ano_data_path = '/data/lq/dataset/cifar/'
    else:
        print('no the anomaly dataset: ',args.anomaly_dataset)
        return

    path_result_source, path_result_train, path_result_test, path_lambda, path_name_lambda = lambda_for_path(args)
    args.data_path = '/data/lq/dataset/'+str(args.dataset)+'/'
    args.result_path = path_result_test
    mkdir(args.result_path)
    args.result_path_normal = path_result_source + 'test/' + 'normal_dataset_' + str(args.dataset) + '/' + path_name_lambda
    mkdir(args.result_path_normal)

    # save model path 
    args.save_model_path = path_result_train + 'saved_models/'
    mkdir(args.save_model_path)

    args.result_test_log_path = args.result_path+'test_log/'
    mkdir(args.result_test_log_path)

    args.result_test_log_path_normal = args.result_path_normal +'test_log/'
    mkdir(args.result_test_log_path_normal)

    # train file
    args.result_train_loss_file = path_result_train + 'loss_train.txt'
    # test files
    # normal test
    args.result_test_loss_file_normal = args.result_path_normal + 'loss_test_normal.txt'
    args.result_measure_acc_true_file = args.result_path_normal + 'measure_acc.txt'
    # anomaly test
    args.result_test_loss_file = args.result_path+'loss_test.txt'
    args.result_measure_prob_file = args.result_path+'measure_prob.txt'
    args.result_measure_norm_file = args.result_path+'measure_norm.txt'
    args.result_measure_recon_file = args.result_path+'measure_recon.txt'
    
    
    # only train init and test normal
    if args.run_state == 'train':
        f = open(args.result_train_loss_file,'w')
        data = 'epoch\t'
        data += 'loss_recon\t'
        data += 'loss_kld\t'
        data += 'loss_ce\t'
        data += 'loss_mem\t'
        data += 'loss_ce_sigmoid\t'
        data += 'loss_norm\n'
        f.write(data)
        f.close()

    if args.run_state == 'train' or args.run_state == 'test_normal':
        f = open(args.result_test_loss_file_normal,'w')
        data = 'epoch\t'
        data += 'loss_normal\t'
        data += 'loss_recon_normal\t'
        data += 'loss_ce_normal\t'
        data += 'loss_mem_normal\t'
        data += 'loss_ce_sigmoid_normal\n'
        f.write(data)
        f.close()

        # f = open(args.result_measure_acc_true_file,'w')
        # f.write('epoch\tacc_true\n')
        # f.close()
    # test all
    elif args.run_state == 'test':

        f = open(args.result_test_loss_file,'w')
        data = 'epoch\t'
        data += 'loss_anomaly\t'
        data += 'loss_recon_anomaly\t'
        data += 'loss_ce_anomaly\t'
        data += 'loss_mem_anomaly\t'
        data += 'loss_ce_sigmoid_anomaly\n'
        f.write(data)
        f.close()
        
        f = open(args.result_measure_prob_file,'w')
        f.write('epoch\ttp\tfp\tfn\ttn\tacc\tauc\tprc\ttnr_at_tpr95\n')
        f.close()
        
        f = open(args.result_measure_norm_file,'w')
        f.write('epoch\ttp\tfp\tfn\ttn\tacc\tauc\tprc\ttnr_at_tpr95\n')
        f.close()
        
        f = open(args.result_measure_recon_file,'w')
        f.write('epoch\ttp\tfp\tfn\ttn\tacc\tauc\tprc\ttnr_at_tpr95\n')
        f.close()


# type_name: norm or prob
# 保存模型
def save_model(multi_classifier, autoencoder, args, epoch):
    save_state = {'autoencoder':autoencoder, 
                'multi_classifier':multi_classifier,
                
                'batch_size':args.batch_size,
                'test_batch_size':args.test_batch_size,
                # 'epochs':args.epochs,
                'lr':args.lr,
                'momentum':args.momentum,
                'use_cuda':args.use_cuda,
                'seed':args.seed,
                'log_interval':args.log_interval,
                'save_model':args.save_model,
                'z_dim':args.z_dim,

                # 'data_path':args.data_path,
                # 'ano_data_path':args.ano_data_path,
                # 'result_path':args.result_path,
                # 'result_test_log_path':args.result_test_log_path,
                # 'result_train_loss_file':args.result_train_loss_file,
                # 'result_test_loss_file':args.result_test_loss_file,
                # 'result_measure_prob_file':args.result_measure_prob_file,
                # 'result_measure_norm_file':args.result_measure_norm_file,
                # 'result_measure_recon_file':args.result_measure_recon_file,
                # 'save_model_path':args.save_model_path,

                'lambda_mem':args.lambda_mem,
                'lambda_ce_sigmoid':args.lambda_ce_sigmoid,
                'lambda_ce':args.lambda_ce,
                'lambda_ae':args.lambda_ae,
                'lambda_norm':args.lambda_norm,

                'dataset':args.dataset,
                # 'anomaly_dataset':args.anomaly_dataset,
                'anomaly_data_number':args.anomaly_data_number,
                'threshold_prob':args.threshold_prob,
                'threshold_norm':args.threshold_norm,
                'threshold_recon':args.threshold_recon,}

    save_model_file_name = args.save_model_path+'/model_epoch_'+str(epoch)+'.pt'
    with open(save_model_file_name, 'wb+') as f:
        torch.save(save_state, save_model_file_name)
    

# type_name: norm or prob
# load 模型
def load_model( args, epoch):
    model_file_name = args.save_model_path+'/model_epoch_'+str(epoch)+'.pt'
    while not os.path.exists(model_file_name):
        print('waiting ... ...')
        time.sleep(360)
    if not os.path.exists(model_file_name):
        return None, None, None
    
    checkpoint = torch.load(model_file_name)
    autoencoder = checkpoint['autoencoder']
    multi_classifier = checkpoint['multi_classifier']
    args.batch_size = checkpoint['batch_size']
    args.test_batch_size = checkpoint['test_batch_size']
    # args.epochs = checkpoint['epochs']
    args.lr = checkpoint['lr']
    args.momentum = checkpoint['momentum']
    args.use_cuda = checkpoint['use_cuda']
    args.seed = checkpoint['seed']
    args.log_interval = checkpoint['log_interval']
    # args.save_model = checkpoint['save_model']
    args.z_dim = checkpoint['z_dim']
    # args.data_path = checkpoint['data_path']
    # args.ano_data_path = checkpoint['ano_data_path']
    # args.result_path = checkpoint['result_path']
    # args.result_test_log_path = checkpoint['result_test_log_path']
    # args.result_train_loss_file = checkpoint['result_train_loss_file']
    # args.result_test_loss_file = checkpoint['result_test_loss_file']
    # args.result_measure_prob_file = checkpoint['result_measure_prob_file']
    # args.result_measure_norm_file = checkpoint['result_measure_norm_file']
    # args.result_measure_recon_file = checkpoint['result_measure_recon_file']
    # args.save_model_path = checkpoint['save_model_path']
    args.lambda_mem = checkpoint['lambda_mem']
    args.lambda_ce_sigmoid = checkpoint['lambda_ce_sigmoid']
    args.lambda_ce = checkpoint['lambda_ce']
    args.lambda_ae = checkpoint['lambda_ae']
    args.lambda_norm = checkpoint['lambda_norm']
    args.dataset = checkpoint['dataset']
    # args.anomaly_dataset = checkpoint['anomaly_dataset']
    args.anomaly_data_number = checkpoint['anomaly_data_number']
    args.threshold_prob = checkpoint['threshold_prob']
    args.threshold_norm = checkpoint['threshold_norm']
    args.threshold_recon = checkpoint['threshold_recon']
    return autoencoder,multi_classifier



# show
# type_name: norm or prob
# 画图 没啥大用
def pred_show(epoch, auc, data, args, type_name):
    data_normal = data[:len(data)-args.anomaly_data_number]
    data_anomaly = data[args.anomaly_data_number:]
    plt.figure()
    plt.hist(data_normal, bins=120, normed=0, facecolor="red", edgecolor="black", alpha=0.7, label='normal')
    plt.hist(data_anomaly, bins=120, normed=0, facecolor="blue", edgecolor="black", alpha=0.7, label='anomaly')
    plt.legend()
    plt.title(type_name+'(auc='+str(auc)+')')
    plt.savefig(args.result_test_log_path+'epoch_'+str(epoch)+'_'+type_name+'.jpg')

    # 在存储一份在汇总文件夹中 /data/lq/result/anomaly_detection_result/out_of_distribution/show_figures/pred_show/
    
    path_result_source, path_result_train, path_result_test, path_lambda, path_name_lambda = lambda_for_path(args)
    path_show = '/data/lq/result/anomaly_detection_result/out_of_distribution/show_figures/pred_show/'
    path_show += path_lambda+'/'
    path_show += type_name+'/'
    mkdir(path_show)
    plt.savefig(path_show+'epoch_'+str(epoch)+'_'+type_name+'.jpg')
    plt.close()

# 画auc图
def auc_show(args):
    path_result_source, path_result_train, path_result_test, path_lambda, path_name_lambda = lambda_for_path(args)
    file_name = 'measure_fig_'+str(args.dataset)+'_'+str(args.anomaly_dataset)+'_'+path_lambda+'.jpg'
    
    df_prob = pd.read_csv(path_result_test+'measure_prob.txt','\t')
    auc_prob = df_prob['auc']
    df_norm = pd.read_csv(path_result_test+'measure_norm.txt','\t')
    auc_norm = df_norm['auc']
    df_recon = pd.read_csv(path_result_test+'measure_recon.txt','\t')
    auc_recon = df_recon['auc']

    plt.figure()
    plt.plot(auc_prob,color='red',label='prob')
    plt.plot(auc_norm,color='blue',label='norm')
    plt.plot(auc_recon,color='green',label='recon')
    plt.title('auc measure')
    plt.legend()
    plt.savefig(path_result_test+file_name)

    path_show = '/data/lq/result/anomaly_detection_result/out_of_distribution/show_figures/auc_show/'
    path_show += path_lambda+'/'
    mkdir(path_show)
    plt.savefig(path_show+file_name)

    plt.close()

# 画loss图
def loss_show(args):
    path_result_source, path_result_train, path_result_test, path_lambda, path_name_lambda = lambda_for_path(args)
    file_name = 'loss_train_fig_'+path_lambda+'.jpg'
    df = pd.read_csv(path_result_train+'loss_train.txt','\t')
    ae = df['loss_recon']
    ce = df['loss_ce']
    mem = df['loss_mem']
    ce_sigmoid = df['loss_ce_sigmoid']
    norm = df['loss_norm']
    plt.figure()
    plt.plot(ae,color='green',label='ae')
    plt.plot(ce,color='yellow',label='ce')
    plt.plot(mem,color='magenta',label='mem')
    plt.plot(ce_sigmoid,color='red',label='ce_sigmoid')
    plt.plot(norm,color='blue',label='norm')
    plt.title('train loss')
    plt.legend()
    plt.savefig(path_result_train+file_name)

    path_show = '/data/lq/result/anomaly_detection_result/out_of_distribution/show_figures/loss_show/'
    path_show += path_lambda+'/'
    mkdir(path_show)
    plt.savefig(path_show+file_name)

    plt.close()

# 画图
# type_name in ['origin','recon_mu', 'recon_z'] : 分别代表 原图、用mu重构图、 用z重构图
def data_fig_save(args, data, type_name, epoch):
    path_result_source, path_result_train, path_result_test, path_lambda, path_name_lambda = lambda_for_path(args)
    path_show = '/data/lq/result/anomaly_detection_result/out_of_distribution/show_figures/data_show/'
    path_show += path_lambda+'/'
    mkdir(path_show)
    file_name = 'data_'+str(epoch)+'_'+str(type_name)+'.jpg'

    if args.dataset in ['mnist']:
        cmap = plt.cm.gray
    else:
        cmap = None
    data = data[:16]
    data = torch.squeeze(data).cpu().detach().numpy()
    if args.dataset not in ['mnist']:
        # 转换到0-1
        # data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data[:,:,:,0] = (data[:,:,:,0] - np.min(data[:,:,:,0])) / (np.max(data[:,:,:,0]) - np.min(data[:,:,:,0]))
        data[:,:,:,1] = (data[:,:,:,1] - np.min(data[:,:,:,1])) / (np.max(data[:,:,:,1]) - np.min(data[:,:,:,1]))
        data[:,:,:,2] = (data[:,:,:,2] - np.min(data[:,:,:,2])) / (np.max(data[:,:,:,2]) - np.min(data[:,:,:,2]))

    grid_length=int(np.ceil(np.sqrt(data.shape[0])))
    plt.figure(figsize=(2*grid_length,2*grid_length))
    gs = gridspec.GridSpec(grid_length,grid_length,wspace=0.1,hspace=0.1)
    for i, img in enumerate(data):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img, cmap = cmap)
        plt.axis('off')
        plt.tight_layout()
    plt.tight_layout()
    plt.savefig(path_show+file_name, bbox_inches='tight')
    plt.close()

# 保存结果
def save_best_model_epoch(args, epoch, type_name, auc, prc, tnr_at_tpr95):
    path_result_source, path_result_train, path_result_test, path_lambda, path_name_lambda = lambda_for_path(args)
    path = path_result_test+'best_model/'
    mkdir(path)
    file_name = 'best_model_'+str(type_name)+'.txt'
    f_best_model = open(path + file_name,'w')
    content = 'epoch ='+str(epoch)+'\n'
    content += type_name+'_auc = '+str(float(round(auc,4)))+'\n'
    content += type_name+'_prc = '+str(float(round(prc,4)))+'\n'
    content += type_name+'_tnr_at_tpr95 = '+str(float(round(tnr_at_tpr95,4)))+'\n'
    f_best_model.write(content)
    f_best_model.close()

# 保存结果
def save_model_test_anomaly(args, epoch, auc_prob, prc_prob, tnr_at_tpr95_prob, max_auc_prob, auc_norm, prc_norm, tnr_at_tpr95_norm, max_auc_norm, auc_recon, prc_recon, tnr_at_tpr95_recon, max_auc_recon, auc_iforest, prc_iforest, tnr_at_tpr95_iforest, max_auc_iforest):
    if round(auc_prob,4) > round(max_auc_prob,4):
        max_auc_prob = auc_prob
        save_best_model_epoch(args, epoch, 'prob', auc_prob, prc_prob, tnr_at_tpr95_prob)
    if round(auc_norm,4) > round(max_auc_norm,4):
        max_auc_norm = auc_norm
        save_best_model_epoch(args, epoch, 'norm', auc_norm, prc_norm, tnr_at_tpr95_norm)
    if round(auc_recon,4) > round(max_auc_recon,4):
        max_auc_recon = auc_recon
        save_best_model_epoch(args, epoch, 'recon', auc_recon, prc_recon, tnr_at_tpr95_recon)
    if round(auc_iforest,4) > round(max_auc_iforest,4):
        max_auc_iforest = auc_iforest
        save_best_model_epoch(args, epoch, 'iforest', auc_iforest, prc_iforest, tnr_at_tpr95_iforest)
    return max_auc_prob, max_auc_norm, max_auc_recon, max_auc_iforest

# 初始化模型 不用了
def init_model_for_train(args):
    device = torch.device("cuda" if args.use_cuda else "cpu")

    if args.dataset == 'mnist': 
        args.z_dim = 10
        multi_classifier = Multi_Classifier_MNIST(args).to(device)
        autoencoder = Auto_Encoder_MNIST(args).to(device)
        # dataset = mnist_dataset(args)
    elif args.dataset == 'cifar': 
        args.z_dim = 20
        multi_classifier = Multi_Classifier_CIFAR(args).to(device)
        autoencoder = Auto_Encoder_CIFAR(args).to(device)
        # dataset = cifar_dataset(args)
    elif args.dataset == 'cifar100': 
        args.z_dim = 200
        multi_classifier = Multi_Classifier_CIFAR100(args).to(device)
        autoencoder = Auto_Encoder_CIFAR100(args).to(device)
        # dataset = cifar100_dataset(args)
    elif args.dataset == 'svhn': 
        args.z_dim = 20
        multi_classifier = Multi_Classifier_SVHN(args).to(device)
        autoencoder = Auto_Encoder_SVHN(args).to(device)
        # dataset = svhn_dataset(args)
    else:
        print('utils.py init_model_for_train: no dataset',args.dataset)
        return
    # train_loader, test_loader_normal = dataset.getdata(args)
    optimizer_multi_classifier = optim.Adam(multi_classifier.parameters(), lr=args.lr)
    optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=args.lr)

    return multi_classifier, autoencoder, optimizer_multi_classifier, optimizer_autoencoder







