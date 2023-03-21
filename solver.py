from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from sklearn import metrics
import matplotlib
import sys
import os
from datetime import datetime

from utils import *
from dataset import *
from model import *




def train(args, multi_classifier, autoencoder, device, train_loader, optimizer_multi_classifier, optimizer_autoencoder, epoch):
    multi_classifier.train()
    autoencoder.train()
    print_grad = False

    # 查看梯度
    grads = {}
    def save_grad(name):
        def hook(grad):
            grads[name] = grad
        return hook


    # add on 2020-02-18 2000 for only ae; 1000 for vae
    only_ae_train_or_test = False
    if args.lambda_ae == 2.0 and args.lambda_ce==0.0 and args.lambda_mem==0.0 and args.lambda_ce_sigmoid==0.0:
        only_ae_train_or_test = True

    f_train_loss = open(args.result_train_loss_file,'a')
    loss_ae_recon_all = 0
    loss_ae_kld_all = 0
    loss_ce_all = 0
    loss_mem_all = 0
    loss_ce_sigmoid_all = 0
    loss_z_norm_all = 0

    de_time = 0
    ae_time = 0
    ce_time = 0

    len_train_loader = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx == 0:
        #     print('train data:',data.shape,data.dtype, '\n', data[0])

        data, target = data.to(device), target.to(device)
        num_classes = 10
        if args.dataset == 'cifar100':
            num_classes = 100
        target_onehot = torch.zeros(target.shape[0], num_classes).scatter_(1, target.view(-1,1).type(torch.LongTensor), 1).to(device)

        mu_logvar = autoencoder.encode(data)
        mu, logvar, z = reparameterizetion(mu_logvar, args.z_dim)
        if only_ae_train_or_test:
            x_hat = autoencoder.decode(mu)
            output = multi_classifier(mu)
        else:
            x_hat = autoencoder.decode(z)
            output = multi_classifier(z)
        # add data show
        if batch_idx == 500:
            x_hat_z = x_hat
            x_hat_mu = autoencoder.decode(mu)
            data_fig_save(args, data.transpose(1,2).transpose(2,3), 'origin', epoch)
            data_fig_save(args, x_hat_z.transpose(1,2).transpose(2,3), 'recon_z', epoch)
            # data_fig_save(args, x_hat_mu.transpose(1,2).transpose(2,3), 'recon_mu', batch_idx)
        # 增加norm训练
        loss_z_norm = torch.norm(z,p=2)
        output_sigmoid = torch.sigmoid(output)
        output_softmax = F.softmax(output, dim=1)

        loss_ce = F.cross_entropy(output, target)
        loss_mem = membership_loss(output_sigmoid, target)
        loss_ce_sigmoid = ce_sigmoid_loss(output_sigmoid, target_onehot)
        # loss_ae_recon = F.binary_cross_entropy(x_hat, data, size_average=False).div(args.batch_size)
        loss_ae_recon = recon_loss(x_hat, data, args.batch_size)
        loss_ae_kld = kl_divergence(mu, logvar)
        loss_vae = loss_ae_recon + loss_ae_kld
        if only_ae_train_or_test:
            loss_vae = loss_ae_recon

        weight_loss_ae = 1.0/(loss_vae.item()+1e-8)
        weight_loss_ce = 1.0/(loss_ce.item()+1e-8)
        weight_loss_mem = 1.0/(loss_mem.item()+1e-8)
        weight_loss_ce_sigmoid = 1.0/(loss_ce_sigmoid.item()+1e-8)

        loss_vae = args.lambda_ae * loss_vae * weight_loss_ae
        loss_multi_classifier = args.lambda_ce * loss_ce * weight_loss_ce
        loss_multi_classifier += args.lambda_mem * loss_mem * weight_loss_mem
        loss_multi_classifier += args.lambda_ce_sigmoid * loss_ce_sigmoid * weight_loss_ce_sigmoid
        
        if args.type_supervised == 'semi':
            if batch_idx < int((args.unlabel_percent)*len_train_loader):
                # 无标签的数据
                # print(batch_idx,'unlabeled')
                loss = loss_vae
            else:
                # 有标签的数据
                # print(batch_idx,'labeled')
                loss = loss_multi_classifier + loss_vae
        elif args.type_supervised == 'supervised':
            if batch_idx < int((1-args.unlabel_percent)*len_train_loader):
                loss = loss_multi_classifier + loss_vae
            else:
                print('batch_idx =',batch_idx, 'len_train_loader =',len_train_loader,'  有监督部分训练完，停止训练')
                break
            


        # 保存梯度        
        if print_grad:
            x_hat.register_hook(save_grad('x_hat'))
            output.register_hook(save_grad('output'))
            z.register_hook(save_grad('z'))


        optimizer_multi_classifier.zero_grad()
        optimizer_autoencoder.zero_grad()
        loss.backward()
        optimizer_multi_classifier.step()
        optimizer_autoencoder.step()

        
        # if print_grad and batch_idx == 500:
        #     # 输出梯度
        #     print('x_hat grad =',grads['x_hat'][0][0][10])
        #     print('output grad =',grads['output'][0])
        #     print('z grad =',grads['z'][0])

        if batch_idx % args.log_interval == (args.log_interval/2):
            # 2020-02-01 change
            # print('weights ae =',round(weight_loss_ae,8),' ce =',round(weight_loss_ce,8),' ce =',round(weight_loss_mem,8),' ce =',round(weight_loss_ce_sigmoid,8))
            print_log1 = 'weights '
            if args.lambda_ae != 0.0:
                print_log1 += 'ae ='+str(round(weight_loss_ae,8))+'\t'
            if args.lambda_ce != 0.0:
                print_log1 += 'ce ='+str(round(weight_loss_ce,8))+'\t'
            if args.lambda_mem != 0.0:
                print_log1 += 'mem ='+str(round(weight_loss_mem,8))+'\t'
            if args.lambda_ce_sigmoid != 0.0:
                print_log1 += 'ce_sigmoid ='+str(round(weight_loss_ce_sigmoid,8))+'\t'
            print(print_log1)

            print_log2 = 'Train Epoch: '+str(epoch)+' ['+str(batch_idx*len(data))+'/'+str(len(train_loader.dataset))+' ('+str(round(100. * batch_idx/len(train_loader),0))+'%)]\t'
            if args.lambda_ae != 0.0:
                print_log2 += 'loss_ae_recon: '+str(round(loss_ae_recon.item(),4))+'\t'
                print_log2 += 'loss_ae_kld: '+str(round(loss_ae_kld.item(),4))+'\t'
            if args.lambda_ce != 0.0:
                print_log2 += 'loss_ce: '+str(round(loss_ce.item(),4))+'\t'
            if args.lambda_mem != 0.0:
                print_log2 += 'loss_mem: '+str(round(loss_mem.item(),4))+'\t'
            if args.lambda_ce_sigmoid != 0.0:
                print_log2 += 'loss_ce_sigmoid: '+str(round(loss_ce_sigmoid.item(),4))+'\t'
            print(print_log2)

        loss_ae_recon_all += round(loss_ae_recon.item(),6)
        loss_ae_kld_all += round(loss_ae_kld.item(),6)
        loss_ce_all += round(loss_ce.item(),6)
        loss_mem_all += round(loss_mem.item(),6)
        loss_ce_sigmoid_all += round(loss_ce_sigmoid.item(),6)
        loss_z_norm_all += round(loss_z_norm.item(),6)
    
    loss_ae_recon_all /= len(train_loader)
    loss_ae_kld_all /= len(train_loader)
    loss_ce_all /= len(train_loader)
    loss_mem_all /= len(train_loader)
    loss_ce_sigmoid_all /= len(train_loader)
    loss_z_norm_all /= len(train_loader)

    f_train_loss.write(str(epoch)+'\t')
    # f_train_loss.write(str(round(loss_ae_recon.item(),6))+'\t')
    # f_train_loss.write(str(round(loss_ae_kld.item(),6))+'\t')
    # f_train_loss.write(str(round(loss_ce.item(),6))+'\t')
    # f_train_loss.write(str(round(loss_mem.item(),6))+'\t')
    # f_train_loss.write(str(round(loss_ce_sigmoid.item(),6))+'\t')
    # f_train_loss.write(str(round(loss_z_norm.item(),6))+'\n')
    f_train_loss.write(str(round(loss_ae_recon_all,6))+'\t')
    f_train_loss.write(str(round(loss_ae_kld_all,6))+'\t')
    f_train_loss.write(str(round(loss_ce_all,6))+'\t')
    f_train_loss.write(str(round(loss_mem_all,6))+'\t')
    f_train_loss.write(str(round(loss_ce_sigmoid_all,6))+'\t')
    f_train_loss.write(str(round(loss_z_norm_all,6))+'\n')
    f_train_loss.close()

    # print('encoder time cost =',de_time)
    # print('decoder time cost =',ae_time)
    # print('ce time cost =',ce_time)


def test_normal(args, multi_classifier, autoencoder,  device, test_loader, epoch):
    multi_classifier.eval()
    autoencoder.eval()

    # add on 2020-02-18 2000 for only ae; 1000 for vae
    only_ae_train_or_test = False
    if args.lambda_ae == 2.0 and args.lambda_ce==0.0 and args.lambda_mem==0.0 and args.lambda_ce_sigmoid==0.0:
        only_ae_train_or_test = True

    # 记录每个epoch的情况  记录每个data point 的预测值和真实标签值 prob norm recon z
    f_prob = open(args.result_test_log_path_normal+'epoch_'+str(epoch)+'.txt','w')
    f_pred_prob = open(args.result_test_log_path_normal+'epoch_'+str(epoch)+'_pred_prob.txt','w')
    f_z_norm = open(args.result_test_log_path_normal+'epoch_'+str(epoch)+'_pred_norm.txt','w')
    f_recon = open(args.result_test_log_path_normal+'epoch_'+str(epoch)+'_pred_recon.txt','w')
    f_z = open(args.result_test_log_path_normal+'epoch_'+str(epoch)+'_z.txt','w')
    f_prob.write('target\tpred\tano_target\tano_pred_prob\tmax_prob\tano_pred_norm\tnorm\tano_pred_recon\trecon\n')
    f_pred_prob.write('prob\tlabel\n')
    f_z_norm.write('norm\tlabel\n')
    f_recon.write('recon\tlabel\n')
    f_z.write('z\tlabel\n')
    
    # 真实的target
    target_list = [] #真实的类别标签
    ano_target_list = [] #异常检测里面的类别标签
    
    # 使用概率值预测使用到的存储
    pred_prob_list = [] #真实的预测类别
    prod_list = [] #最大概率值
    ano_pred_prob_list = [] #异常检测里面的预测类别 prob
    

    # 使用norm做预测使用到的存储
    norm_list = [] #真实的norm结果
    ano_norm_list = [] #用于异常检测的norm结果，做 负归一化
    ano_pred_norm_list = [] #异常检测里面的预测类别 norm

    # 使用recon做预测使用到的存储
    recon_list = [] #真实的recon结果
    ano_recon_list = [] #用于异常检测的recon结果，做 负归一化
    ano_pred_recon_list = [] #异常检测里面的预测类别 recon

    with torch.no_grad():
        # normal test data
        test_loss_ae_normal = 0
        test_loss_ce_normal = 0
        test_loss_mem_normal = 0
        test_loss_ce_sigmoid_normal = 0
        
        for test_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            num_classes = 10
            if args.dataset == 'cifar100':
                num_classes = 100
            target_onehot = torch.zeros(target.shape[0], num_classes).scatter_(1, target.view(-1,1).type(torch.LongTensor), 1).to(device)
            mu_logvar = autoencoder.encode(data)
            mu, logvar, z = reparameterizetion(mu_logvar, args.z_dim)
            x_hat = autoencoder.decode(mu)
            output = multi_classifier(mu)
            output_sigmoid = torch.sigmoid(output)
            output_softmax = F.softmax(output, dim=1)

            test_loss_ae_normal += recon_loss(x_hat, data, 1).item()
            test_loss_ce_normal += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            test_loss_mem_normal += membership_loss(output_sigmoid, target).item()
            test_loss_ce_sigmoid_normal += ce_sigmoid_loss(output_sigmoid, target_onehot).item()
            
            # 开始计算test效果
            target_list.append(int(target[0].item()))
            ano_target = 1
            ano_target_list.append(ano_target)
            # 
            # 使用prob判别异常
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            pred_prob_list.append(int(pred[0][0].item()))
            # 不用membership和多标签，则使用softmax作为区分
            if args.lambda_mem == 0 and args.lambda_ce_sigmoid == 0:
                pred_prob = float(output_softmax[0][pred[0][0]])  
            else:
                pred_prob = float(output_sigmoid[0][pred[0][0]])
            prod_list.append(pred_prob)
            if pred_prob>= args.threshold_prob:
                ano_pred = 1
            else:
                ano_pred = 0
            ano_pred_prob_list.append(ano_pred)
            
            # 
            # 使用norm判别异常
            z_norm = min(10000, torch.norm(z,p=2).item())
            if z_norm > args.threshold_norm:
                ano_pred_norm = 0
            else:
                ano_pred_norm = 1
            ano_pred_norm_list.append(ano_pred_norm)
            norm_list.append(z_norm)
            # 
            # 使用recon判别异常
            data_recon = recon_loss(x_hat, data, 1).item() 
            if not only_ae_train_or_test:
                data_recon += kl_divergence(mu, logvar).item() #加入kld作为recon一起
            if data_recon > args.threshold_recon:
                ano_pred_recon = 0
            else:
                ano_pred_recon = 1
            ano_pred_recon_list.append(ano_pred_recon)
            recon_list.append(data_recon)

            # test log 记录信息
            f_prob.write(str(int(target))+'\t'+str(int(pred))+'\t'+str(int(ano_target))+'\t'+str(int(ano_pred))+'\t'+str(round(pred_prob,6))+'\t'+str(int(ano_pred_norm))+'\t'+str(round(z_norm,6))+'\t'+str(int(ano_pred_recon))+'\t'+str(round(data_recon,6))+'\n')
            f_pred_prob.write(str(pred_prob)+'\t'+str(1)+'\n') # 记录每个数据点的prob结果
            f_z_norm.write(str(z_norm)+'\t'+str(1)+'\n') # 记录每个数据点的norm结果
            f_recon.write(str(data_recon)+'\t'+str(1)+'\n') # 记录每个数据点的recon结果
            # f_z.write(str(','.join([str(z_i.item()) for z_i in z[0]])) +'\t'+str(1)+'\n')

        test_loss_ae_normal /= len(test_loader.dataset)
        test_loss_ce_normal /= len(test_loader.dataset)
        test_loss_mem_normal /= len(test_loader.dataset)
        test_loss_ce_sigmoid_normal /= len(test_loader.dataset)
        
        test_loss_normal = args.lambda_ae*test_loss_ae_normal + args.lambda_ce*test_loss_ce_normal + args.lambda_mem*test_loss_mem_normal + args.lambda_ce_sigmoid*test_loss_ce_sigmoid_normal
        
        # acc for true label (multi-classifier)
        acc_multiclass = metrics.accuracy_score(target_list, pred_prob_list)
        print('Normal test set: Average loss: {:.4f}, multiclass Accuracy: {:.4f}'.format(test_loss_normal, acc_multiclass))


    # 记录test loss normal
    f_test_loss_normal = open(args.result_test_loss_file_normal,'a')
    f_test_loss_normal.write(str(epoch)+'\t')
    f_test_loss_normal.write(str(round(test_loss_normal,6))+'\t')
    f_test_loss_normal.write(str(round(test_loss_ae_normal,6))+'\t')
    f_test_loss_normal.write(str(round(test_loss_ce_normal,6))+'\t')
    f_test_loss_normal.write(str(round(test_loss_mem_normal,6))+'\t')
    f_test_loss_normal.write(str(round(test_loss_ce_sigmoid_normal,6))+'\n')
    f_test_loss_normal.close()

    f_measure_acc_normal = open(args.result_measure_acc_true_file, 'a')
    f_measure_acc_normal.write(str(epoch)+'\t')
    f_measure_acc_normal.write(str(round(acc_multiclass,6))+'\n')
    f_measure_acc_normal.close()



    f_prob.close()
    f_pred_prob.close()
    f_z_norm.close()
    f_recon.close()
    f_z.close()






def test_anomaly(args, multi_classifier, autoencoder,  device, test_loader_anomaly, epoch):
    
    multi_classifier.eval()
    autoencoder.eval() 
    # 将之前训练好的normal数据读出来 , 一个时到文件中，还有到list中
    ano_target_list, prod_list, ano_pred_prob_list, norm_list, ano_pred_norm_list, recon_list, ano_pred_recon_list = load_normal_test_data(args, epoch)

    only_ae_train_or_test = False
    if args.lambda_ae == 2.0 and args.lambda_ce==0.0 and args.lambda_mem==0.0 and args.lambda_ce_sigmoid==0.0:
        only_ae_train_or_test = True
    
    # if args.run_state == 'test':
    # 记录每个epoch的情况
    f_prob = open(args.result_test_log_path+'epoch_'+str(epoch)+'.txt','a')
    f_pred_prob = open(args.result_test_log_path+'epoch_'+str(epoch)+'_pred_prob.txt','a')
    f_z_norm = open(args.result_test_log_path+'epoch_'+str(epoch)+'_pred_norm.txt','a')
    f_recon = open(args.result_test_log_path+'epoch_'+str(epoch)+'_pred_recon.txt','a')
    f_z = open(args.result_test_log_path+'epoch_'+str(epoch)+'_z.txt','a')

    with torch.no_grad():
        test_loss_autoencoder_anomaly = 0
        test_loss_ce_anomaly = 0
        test_loss_mem_anomaly = 0
        test_loss_ce_sigmoid_anomaly = 0

        test_idx_sum = len(test_loader_anomaly)
        for test_idx, (data, target) in enumerate(test_loader_anomaly):
            # if args.anomaly_dataset == 'svhn':
            #     if args.dataset == 'mnist':
            #         data = data[:,0,:28,:28].view(1,1,28,28)
            #     else:
            #         data = data[:,:,:32,:32].view(1,3,32,32)
            if args.dataset == 'mnist':
                data = data[:,0,:28,:28].view(1,1,28,28)
                # print(data.min(), data.max())
                # data = (data-data.min())/(data.max()-data.min())
            

            data, target = data.type(torch.FloatTensor).to(device), target.to(device)
            num_classes = 10
            if args.dataset == 'cifar100':
                num_classes = 100
            target_onehot = torch.zeros(target.shape[0], num_classes).scatter_(1, target.view(-1,1).type(torch.LongTensor), 1).to(device)

            mu_logvar = autoencoder.encode(data)
            mu, logvar, z = reparameterizetion(mu_logvar, args.z_dim)
            if args.lambda_ae != 0:
                x_hat = autoencoder.decode(mu)
            output = multi_classifier(mu)
            output_sigmoid = torch.sigmoid(output)
            output_softmax = F.softmax(output, dim=1)

            test_loss_mem_anomaly += membership_loss(output_sigmoid, target).item()
            test_loss_ce_anomaly += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            if args.lambda_ae != 0:
                test_loss_autoencoder_anomaly += min(10000, recon_loss(x_hat, data, 1).item())
            test_loss_ce_sigmoid_anomaly += ce_sigmoid_loss(output_sigmoid, target_onehot).item()
            # 开始计算test效果
            ano_target = target
            ano_target_list.append(int(ano_target[0].item()))
            # 
            # 使用概率值判别异常
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            # 不用membership和多标签，则使用softmax作为区分
            if args.lambda_mem == 0 and args.lambda_ce_sigmoid == 0:
                pred_prob = float(output_softmax[0][pred[0][0]])  
            else:
                pred_prob = float(output_sigmoid[0][pred[0][0]])
            prod_list.append(pred_prob)
            if pred_prob>= args.threshold_prob:
                ano_pred = 1
            else:
                ano_pred = 0
            ano_pred_prob_list.append(ano_pred)
            
            # 
            # 使用norm判别异常
            z_norm = min(1e6, torch.norm(z,p=2).item())
            if z_norm > args.threshold_norm:
                ano_pred_norm = 0
            else:
                ano_pred_norm = 1
            ano_pred_norm_list.append(ano_pred_norm)
            norm_list.append(z_norm)
            # 
            # 使用recon判别异常
            if args.lambda_ae != 0:
                data_recon = recon_loss(x_hat, data, 1).item()
                if not only_ae_train_or_test:
                    data_recon += kl_divergence(mu, logvar).item() #加入kld作为recon一起
                data_recon = min(1e6, data_recon)
                if data_recon > args.threshold_recon:
                    ano_pred_recon = 0
                else:
                    ano_pred_recon = 1
                ano_pred_recon_list.append(ano_pred_recon)
                recon_list.append(data_recon)

            # test log 记录信息
            # if args.run_state == 'test':
            data_prob = str(int(-1))+'\t'+str(int(pred))+'\t'+str(int(ano_target))
            data_prob += '\t'+str(int(ano_pred))+'\t'+str(round(pred_prob,6))
            data_prob += '\t'+str(int(ano_pred_norm))+'\t'+str(round(z_norm,6))
            if args.lambda_ae != 0:
                data_prob += '\t'+str(int(ano_pred_recon))+'\t'+str(round(data_recon,6))
            else:
                data_prob += '\t'+str(int(-1))+'\t'+str(round(-1,6))
            data_prob += '\n'
            f_prob.write(data_prob)
            f_pred_prob.write(str(pred_prob)+'\t'+str(0)+'\n')  # 记录每个数据点的prob结果
            f_z_norm.write(str(z_norm)+'\t'+str(0)+'\n')   # 记录每个数据点的norm结果
            if args.lambda_ae != 0:
                f_recon.write(str(data_recon)+'\t'+str(0)+'\n') # 记录每个数据点的recon结果
            # f_z.write(str(','.join([str(z_i.item()) for z_i in z[0]])) +'\t'+str(0)+'\n')

        test_loss_mem_anomaly /= len(test_loader_anomaly.dataset)
        test_loss_ce_anomaly /= len(test_loader_anomaly.dataset)
        test_loss_autoencoder_anomaly /= len(test_loader_anomaly.dataset)
        test_loss_ce_sigmoid_anomaly /= len(test_loader_anomaly.dataset)
        
        test_loss_anomaly = args.lambda_ae*test_loss_autoencoder_anomaly + args.lambda_ce*test_loss_ce_anomaly + args.lambda_mem*test_loss_mem_anomaly + args.lambda_ce_sigmoid*test_loss_ce_sigmoid_anomaly
        print('Anomaly test set: Average loss: {:.4f}'.format(test_loss_anomaly))
    

    # 
    # prob measure
    auc_prob = metrics.roc_auc_score(ano_target_list, prod_list)
    precision_prob, recall_prob, _thresholds = metrics.precision_recall_curve(ano_target_list, prod_list)
    prc_prob = metrics.auc(recall_prob, precision_prob)
    tp_prob, fn_prob, fp_prob, tn_prob = confusion_matrix(ano_target_list, ano_pred_prob_list)
    acc_ano_prob = (tp_prob+tn_prob)/(1.0*(tp_prob+fn_prob+fp_prob+tn_prob))
    # tnr_at_tpr95
    tnr_at_tpr95_prob = tnr_at_tpr95_cal(ano_target_list, prod_list)
    print('\nauc_prob =',round(auc_prob,4),'\tprc_prob =',round(prc_prob,4),'\tnr_at_tpr95_prob =',round(tnr_at_tpr95_prob,4))
    pred_show(epoch=epoch, auc=auc_prob, data=prod_list, args=args, type_name='prob')
    
    # 
    # norm measure
    # 进行负归一化norm
    max_norm = max(norm_list)
    min_norm = min(norm_list)
    ano_norm_list = [ (max_norm-i)/(max_norm-min_norm) for i in norm_list]
    
    auc_norm = metrics.roc_auc_score(ano_target_list, ano_norm_list)
    precision_norm, recall_norm, _thresholds = metrics.precision_recall_curve(ano_target_list, ano_norm_list)
    prc_norm = metrics.auc(recall_norm, precision_norm)
    tp_norm, fn_norm, fp_norm, tn_norm = confusion_matrix(ano_target_list, ano_pred_norm_list)
    acc_ano_norm = (tp_norm+tn_norm)/(1.0*(tp_norm+fn_norm+fp_norm+tn_norm))
    # tnr_at_tpr95
    tnr_at_tpr95_norm = tnr_at_tpr95_cal(ano_target_list, norm_list)
    print('\nauc_norm =',round(auc_norm,4),'\tprc_norm =',round(prc_norm,4),'\ttnr_at_tpr95_norm =',round(tnr_at_tpr95_norm,4))
    pred_show(epoch=epoch, auc=auc_norm, data=norm_list, args=args, type_name='norm')

    # 
    # recon measure
    # 进行负归一化recon
    if args.lambda_ae != 0:
        max_recon = max(recon_list)
        min_recon = min(recon_list)
        ano_recon_list = [ (max_recon-i)/(max_recon-min_recon) for i in recon_list]
    
        auc_recon = metrics.roc_auc_score(ano_target_list, ano_recon_list)
        precision_recon, recall_recon, _thresholds = metrics.precision_recall_curve(ano_target_list, ano_recon_list)
        prc_recon = metrics.auc(recall_recon, precision_recon)
        tp_recon, fn_recon, fp_recon, tn_recon = confusion_matrix(ano_target_list, ano_pred_recon_list)
        acc_ano_recon = (tp_recon+tn_recon)/(1.0*(tp_recon+fn_recon+fp_recon+tn_recon))
        # tnr_at_tpr95
        tnr_at_tpr95_recon = tnr_at_tpr95_cal(ano_target_list, recon_list)
        print('\nauc_recon =',round(auc_recon,4),'\tprc_recon =',round(prc_recon,4),'\ttnr_at_tpr95_recon =',round(tnr_at_tpr95_recon,4))

        pred_show(epoch=epoch, auc=auc_recon, data=recon_list, args=args, type_name='recon')
    
    # if args.run_state == 'test':
    f_test_loss = open(args.result_test_loss_file,'a')
    f_measure_prob = open(args.result_measure_prob_file,'a')
    f_measure_norm = open(args.result_measure_norm_file,'a')
    if args.lambda_ae != 0:
        f_measure_recon = open(args.result_measure_recon_file,'a')
    
    # 记录test loss
    f_test_loss.write(str(epoch)+'\t')
    f_test_loss.write(str(round(test_loss_anomaly,6))+'\t')
    f_test_loss.write(str(round(test_loss_autoencoder_anomaly,6))+'\t')
    f_test_loss.write(str(round(test_loss_ce_anomaly,6))+'\t')
    f_test_loss.write(str(round(test_loss_mem_anomaly,6))+'\t')
    f_test_loss.write(str(round(test_loss_ce_sigmoid_anomaly,6))+'\n')

    # 记录measure的结果
    f_measure_prob.write(str(epoch)+'\t'+str(tp_prob)+'\t'+str(fp_prob)+'\t'+str(fn_prob)+'\t'+str(tn_prob)+'\t'+str(round(acc_ano_prob,4))+'\t'+str(round(auc_prob,4))+'\t'+str(round(prc_prob,4))+'\t'+str(round(tnr_at_tpr95_prob,4))+'\n')
    f_measure_norm.write(str(epoch)+'\t'+str(tp_norm)+'\t'+str(fp_norm)+'\t'+str(fn_norm)+'\t'+str(tn_norm)+'\t'+str(round(acc_ano_norm,4))+'\t'+str(round(auc_norm,4))+'\t'+str(round(prc_norm,4))+'\t'+str(round(tnr_at_tpr95_norm,4))+'\n')
    if args.lambda_ae != 0:
        f_measure_recon.write(str(epoch)+'\t'+str(tp_recon)+'\t'+str(fp_recon)+'\t'+str(fn_recon)+'\t'+str(tn_recon)+'\t'+str(round(acc_ano_recon,4))+'\t'+str(round(auc_recon,4))+'\t'+str(round(prc_recon,4))+'\t'+str(round(tnr_at_tpr95_recon,4))+'\n')

    f_test_loss.close()
    f_measure_prob.close()
    f_measure_norm.close()
    if args.lambda_ae != 0:
        f_measure_recon.close()

    f_prob.close()
    f_pred_prob.close()
    f_z_norm.close()
    f_recon.close()
    f_z.close()
    
    if args.lambda_ae == 0:
        auc_recon, prc_recon, tnr_at_tpr95_recon = 0.0, 0.0, 0.0

    return auc_prob, prc_prob, tnr_at_tpr95_prob, auc_norm, prc_norm, tnr_at_tpr95_norm, auc_recon, prc_recon, tnr_at_tpr95_recon



