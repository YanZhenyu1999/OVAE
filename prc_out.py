import pandas as pd 
from pandas.core.frame import DataFrame
import numpy as np
from sklearn import metrics
import sys
import os
# def tnr_at_tpr95_cal(ano_target_list, prob_list):
#     if len(ano_target_list) < 20000:
#         return 0
#     positive_prod_list = prob_list[:10000]
#     positive_prod_list.sort()
#     threshold = positive_prod_list[500]
#     pred_list = []
#     for i in range(len(prob_list)):
#         if prob_list[i] >= threshold:
#             pred_list.append(1)
#         else:
#             pred_list.append(0)
#     true_negative = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(ano_target_list, pred_list))
#     actual_negative = len(ano_target_list) - sum(ano_target_list)
#     tnr = true_negative / actual_negative
#     return tnr

# def fpr95_cal(labels, scores):
#     recall_point = 0.9
#     labels = np.asarray(labels)
#     scores = np.asarray(scores)
#     # Sort label-score tuples by the score in descending order.
#     indices = np.argsort(scores)[::-1]    #降序排列
#     sorted_labels = labels[indices]
#     sorted_scores = scores[indices]
#     n_match = sum(sorted_labels)
#     n_thresh = recall_point * n_match
#     thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
#     FP = np.sum(sorted_labels[:thresh_index] == 0)
#     TN = np.sum(sorted_labels[thresh_index:] == 0)
#     return float(FP) / float(FP + TN)


def prc_out_cal(ano_target_list, prob_list):
    label_out = [-i+1 for i in ano_target_list]
    prob_out = [-i for i in prob_list]
    precision_prob, recall_prob, _thresholds = metrics.precision_recall_curve(label_out, prob_out)
    prc_prob = metrics.auc(recall_prob, precision_prob)
    return prc_prob
# dataset = 'cifar'
# anomaly_dataset = 'svhn'
# net_type = 'densenet121'
# lambda_ae = 0.0
# lambda_ce = 1.0
# lambda_ce_sigmoid = 0.0
# type_name = 'prob'
# path = '/data/lq/result/anomaly_detection_result/out_of_distribution/'+str(dataset)+'/'
# path += 'test/anomaly_dataset_' + str(anomaly_dataset)
# path += '/net_type_'+str(net_type)+'/'
# path += 'lambda_ae_'+str(lambda_ae)+'/'
# path += 'lambda_ce_'+str(lambda_ce)+'/'
# path += 'lambda_mem_'+str(lambda_mem)+'/'
# path += 'lambda_ce_sigmoid_'+str(lambda_ce_sigmoid)+'/'
# print(path)
# path_epoch = path+'test_log/'


lambda_mem = 0
# ['mnist', 'cifar', 'cifar100', 'svhn']
dataset_list = ['mnist']
# ['gaussian','uniform', 'cifar10','cifar100','svhn','imagenet','lsun']
anomaly_dataset_list = ['cifar10']
# ['mlp', 'resnet34', 'densenet121']
net_type_list = ['mlp']
lambda_ae_list = [0]
lambda_ce_list = [1]
lambda_ce_sigmoid_list = [0]
type_name_list = ['prob']
# ['prob', 'norm', 'recon']
# ['mnist', 'cifar', 'cifar100', 'svhn']
for dataset in dataset_list:
    path = '/data/lq/result/anomaly_detection_result/out_of_distribution/'+str(dataset)+'/'
    if not os.path.exists(path):
        print(path,'not exists')
        continue
    for anomaly_dataset in anomaly_dataset_list:
        path1= path+'test/anomaly_dataset_' + str(anomaly_dataset)+'/'
        if not os.path.exists(path1):
            print(path1,'not exists')
            continue
        # print(path1)
        for net_type in net_type_list:
            path2= path1+'net_type_'+str(net_type)+'/'
            if not os.path.exists(path2):
                print(path2,'not exists')
                continue
            # print(path2)
            for lambda_ae in lambda_ae_list:
                path3= path2+'lambda_ae_'+str(float(lambda_ae))+'/'
                if not os.path.exists(path3):
                    print(path3,'not exists')
                    continue
                # print(path3)
                for lambda_ce in lambda_ce_list:
                    path4= path3+'lambda_ce_'+str(float(lambda_ce))+'/'
                    if not os.path.exists(path4):
                        print(path4,'not exists')
                        continue
                    # print(path4)
                    path4 += 'lambda_mem_'+str(float(lambda_mem))+'/'
                    for lambda_ce_sigmoid in lambda_ce_sigmoid_list:
                        path5= path4+'lambda_ce_sigmoid_'+str(float(lambda_ce_sigmoid))+'/'
                        if not os.path.exists(path5):
                            print(path5,'not exists')
                            continue
                        # print(path5)
                        path_epoch = path5+'test_log/'
                        for type_name in type_name_list:
                            if type_name=='recon' and lambda_ae==0:
                                continue
                            file_measure = path5+'measure_'+type_name+'.txt'
                            prc_out = []
                            if not os.path.exists(file_measure):
                                print(file_measure,'not exists')
                                continue
                            # print(file_measure)
                            print(dataset, anomaly_dataset, net_type, lambda_ae, lambda_ce, lambda_mem, lambda_ce_sigmoid, type_name)
                            df_measure = pd.read_csv(file_measure,'\t')
                            if 'prc_out' in df_measure.columns:
                                print('prc_out already in ',file_measure)
                                df_measure_change = df_measure
                            else:
                                for epoch in range(1,301):
                                    if epoch%100==0:
                                        print('epoch :',epoch)
                                    file_epoch = path_epoch+'epoch_'+str(int(epoch))+'.txt'
                                    if not os.path.exists(file_epoch):
                                        # print(file_epoch,'not exists')
                                        continue
                                    df = pd.read_csv(file_epoch,'\t')
                                    ano_target_list = list(df['ano_target'])
                                    if type_name == 'prob':
                                        pred_list = list(df['max_prob'])
                                    else:
                                        pred_list = list(df[type_name])
                                    res_prc_out = prc_out_cal(ano_target_list, pred_list)
                                    prc_out.append(res_prc_out)
                                    prc_out_df=DataFrame({'prc_out':prc_out})
                                    df_measure_change = pd.concat([df_measure,prc_out_df], axis=1)
                                    df_measure_change.to_csv(file_measure,'\t',index=0)

                            file_best_model = path5+'best_model/best_model_'+str(type_name)+'.txt'
                            if not os.path.exists(file_best_model):
                                print(file_best_model,'not exists')
                                continue
                            f=open(file_best_model,'r')
                            epoch_max_auc = int(f.readline()[7:])
                            f.close()
                            # print('type_name =',type_name)
                            # print('epoch =',epoch_max_auc)
                            # print('auc = ',df_measure_change['auc'][epoch_max_auc-1])
                            # print('prc = ',df_measure_change['prc'][epoch_max_auc-1])
                            # print('tnr_at_tpr95 = ',df_measure_change['tnr_at_tpr95'][epoch_max_auc-1])
                            f=open(file_best_model,'r')
                            content = ''.join(f.readlines()[:4])
                            f.close()
                            f=open(file_best_model,'w')
                            content += type_name+'_prc_out = '+str(df_measure_change['prc_out'][epoch_max_auc-1])+'\n'
                            f.write(content)
                            f.close()







