from sklearn.ensemble import IsolationForest
import os
import sys
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, 
import warnings
warnings.simplefilter("ignore")
from utils import tnr_at_tpr95_cal

def pred_transform(data):
    for i in range(data.shape[0]):
        if data[i] == 0:
            data[i] = -1
    return data

def prc_out_cal(ano_target_list, prob_list):
    label_out = [-i for i in ano_target_list]
    prob_out = [-i for i in prob_list]
    precision_prob, recall_prob, _thresholds = metrics.precision_recall_curve(label_out, prob_out)
    prc_prob = metrics.auc(recall_prob, precision_prob)
    return prc_prob

def iforest_func(train_data, test_data, target):
    # iforest
    iforest = IsolationForest(contamination=0.1, max_samples=256, random_state=np.random.RandomState(42))
    iforest.fit(train_data)
    test_pred_iforest = iforest.predict(test_data)
    test_pred_prob_iforest = iforest.decision_function(test_data)
    auc_iforest = metrics.roc_auc_score(target, test_pred_prob_iforest)
    precision_iforest, recall_iforest, _thresholds = metrics.precision_recall_curve(target, test_pred_prob_iforest)
    prc_iforest = metrics.auc(recall_iforest, precision_iforest)
    tnr_at_tpr95_iforest = tnr_at_tpr95_cal(target, test_pred_prob_iforest)
    # print('iforest: auc=',auc_iforest)
    prc_out = prc_out_cal(target, test_pred_prob_iforest)
    return auc_iforest, prc_iforest, tnr_at_tpr95_iforest, prc_out


def pre_data(path, filename, train_data_samples):
    # prob norm recon
    df = pd.read_csv(path + filename,'\t')
    # print(df.columns)
    prob = np.array(df['max_prob']).reshape([-1,1])
    norm = np.array(df['norm']).reshape([-1,1])
    recon = np.array(df['recon']).reshape([-1,1])
    all_target = np.array(df['ano_target'])
    all_target = pred_transform(all_target)

    # ocsvm train test data 
    all_data = np.append(prob, recon, axis=1)
    # test_data = np.append(test_data, norm, axis=1)
    # test_data = np.append(test_data, z, axis=1)
    train_data = all_data[:train_data_samples]
    train_target = all_target[:train_data_samples]

    test_data = all_data[train_data_samples:]
    test_target = all_target[train_data_samples:]
    return train_data, train_target, test_data, test_target

# 初始化路径参数，单独为iforest写的，也很重要，路径地址
def init_path_iforest(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent, type_supervised):
    path = '/data/lq/result/anomaly_detection_result/out_of_distribution/'
    path += dataset+'/test/anomaly_dataset_'+anomaly_dataset+'/'
    path += 'net_type_'+str(net_type)
    path += '/lambda_ae_'+str(ae)
    path += '/lambda_ce_'+str(ce)
    path += '/lambda_mem_'+str(mem)
    path += '/lambda_ce_sigmoid_'+str(ce_sig)
    if type_supervised == 'semi':
        path += '/unlabel_percent_'+str(unlabel_percent)+'/'
    elif type_supervised == 'supervised':
        path += '/supervised_without_unlabel_percent_'+str(unlabel_percent)+'/'
    
    source_path = path+'test_log/'
    res_file = path+'measure_iforest.txt'
    return path, source_path, res_file

def save_best_result(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent, type_supervised):
    path, source_path, res_file = init_path_iforest(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent, type_supervised)
    # 计算每个最大的auc结果
    df = pd.read_csv(res_file,'\t')
    auc_iforest = df['auc']
    prc_iforest = df['prc']
    tnr_at_tpr95_iforest = df['tnr_at_tpr95']
    prc_out_iforest = df['prc_out']
    max_auc_epoch = auc_iforest.idxmax()
    print('epoch =',max_auc_epoch+1)
    print('iforest_auc =' ,round(auc_iforest[max_auc_epoch],4), 
        '\niforest_prc =' ,round(prc_iforest[max_auc_epoch],4),
        '\niforest_tnr_at_tpr95 =' ,round(tnr_at_tpr95_iforest[max_auc_epoch],4),
        '\niforest_prc_out =' ,round(prc_out_iforest[max_auc_epoch],4))
    
    best_res_file = path+'best_model/best_model_iforest.txt'
    f = open(best_res_file,'w')
    f.write('epoch ='+str(max_auc_epoch+1))
    f.write('\niforest_auc = '+str(round(auc_iforest[max_auc_epoch],4)))
    f.write('\niforest_prc = '+str(round(prc_iforest[max_auc_epoch],4)))
    f.write('\niforest_tnr_at_tpr95 = '+str(round(tnr_at_tpr95_iforest[max_auc_epoch],4)))
    f.write('\niforest_prc_out = '+str(round(prc_out_iforest[max_auc_epoch],4)))
    f.write('\n')
    f.close()


# 在main.py中被调用的函数，最为重要
def iforest_test_epoch(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent, epoch, type_supervised):
    train_data_samples = 500
    path, source_path, res_file = init_path_iforest(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent, type_supervised)
    # print('iforest: ',source_path)
    # print('iforest: ',res_file)

    if not os.path.exists(res_file):
        f = open(res_file, 'w')
        f.write('epoch\tauc\tprc\ttnr_at_tpr95\tprc_out\n')
        f.close()
    
    filename = 'epoch_'+str(int(epoch))+'.txt'
    if not os.path.exists(source_path+filename):
        return 0,0,0
    train_data, train_target, test_data, target = pre_data(source_path, filename, train_data_samples)
    auc_iforest, prc_iforest, tnr_at_tpr95_iforest, prc_out_iforest = iforest_func(train_data, test_data, target)
    # write result file 
    f = open(res_file, 'a')
    content = str(int(epoch))+'\t' 
    content += str(round(auc_iforest,4))+'\t' 
    content += str(round(prc_iforest,4))+'\t'
    content += str(round(tnr_at_tpr95_iforest,4))+'\t'
    content += str(round(prc_out_iforest,4))+'\n'
    f.write(content)
    f.close()
    return auc_iforest, prc_iforest, tnr_at_tpr95_iforest, prc_out_iforest

def iforest_test(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent, type_supervised):
    print('iforest test')
    for epoch in range(1, 301):
        if epoch%50==0:
            print('iforest epoch =',epoch, str(epoch/300*100)+'%')
        iforest_test_epoch(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent, epoch, type_supervised)
    save_best_result(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent, type_supervised)




def main():
    dataset = 'svhn'
    anomaly_dataset = 'cifar10'
    net_type = 'densenet201'
    ae = 1.0
    ce = 0.0
    mem = 0.0
    ce_sig = 1.0
    if len(sys.argv) < 10:
        print('state, dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent')
        return
    _, state, dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent = sys.argv
    ae, ce, mem, ce_sig = float(ae), float(ce), float(mem), float(ce_sig), 
    print(state, dataset, anomaly_dataset, ae, ce, mem, ce_sig, unlabel_percent)
    path, source_path, res_file = init_path_iforest(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent)
    if not os.path.exists(path):
        print(path, 'not exists')
        return
    # train
    if state == 'test':
        iforest_test(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent)
    save_best_result(dataset, anomaly_dataset, net_type, ae, ce, mem, ce_sig, unlabel_percent)

if __name__ == '__main__':
    main()