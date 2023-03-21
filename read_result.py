import sys
import os

def readres(dataset, dataset_anomaly, type_name, net_type, lambda_ae, lambda_ce, lambda_mem, lambda_ce_sigmoid, type_supervised, unlabel_percent):
    path = '/data/lq/result/anomaly_detection_result/out_of_distribution/'
    path += str(dataset)
    path += '/test/anomaly_dataset_'+str(dataset_anomaly)
    path += '/net_type_'+str(net_type)
    path += '/lambda_ae_'+str(float(lambda_ae))
    path += '/lambda_ce_'+str(float(lambda_ce))
    path += '/lambda_mem_'+str(float(lambda_mem))
    path += '/lambda_ce_sigmoid_'+str(float(lambda_ce_sigmoid))
    if type_supervised == 'semi':
        path += '/unlabel_percent_'+str(float(unlabel_percent))
    elif type_supervised == 'supervised':
        path += '/supervised_without_unlabel_percent_'+str(float(unlabel_percent))
    path += '/best_model/'
    
    data = ''
    if os.path.isfile(path+'best_model_prob.txt'):
        data_prob = 'prob\n'+''.join(open(path+'best_model_prob.txt','r').readlines()[:3])
    else:
        data_prob = 'prob\n'
    
    # if os.path.isfile(path+'best_model_norm.txt'):
    #     data_norm = 'norm\n'+''.join(open(path+'best_model_norm.txt','r').readlines())
    # else:
    #     data_norm = 'norm\n'
    
    if os.path.isfile(path+'best_model_recon.txt'):
        data_recon = 'recon\n'+''.join(open(path+'best_model_recon.txt','r').readlines()[:3])
    else:
        data_recon = 'recon\n'

    if os.path.isfile(path+'best_model_iforest.txt'):
        data_iforest = 'iforest\n'+''.join(open(path+'best_model_iforest.txt','r').readlines()[:3])
    else:
        data_iforest = 'iforest\n'

    if type_name == 'prob':
        return data_prob
    # elif type_name == 'norm':
    #     return data_norm
    elif type_name == 'recon':
        return data_recon
    elif type_name == 'iforest':
        return data_iforest
    elif type_name == 'all':
        # return data_prob+'\n'+data_recon+'\n'+data_iforest
        return data_prob+'\n'+data_iforest

def main():
    if len(sys.argv)!=11 and len(sys.argv)!=4:
        print('input is wrong, the right lambdas: ')
        print('all input => type_name, net_type, lambda_ae, lambda_ce, lambda_mem, lambda_ce_sigmoid, dataset, dataset_anomaly, type_supervised, unlabel_percent\n')
        print('sample input => model, type_supervised, unlabel_percent')
        return
    if len(sys.argv) == 11:
        _, type_name, net_type, lambda_ae, lambda_ce, lambda_mem, lambda_ce_sigmoid, dataset, dataset_anomaly, type_supervised, unlabel_percent = sys.argv
    elif len(sys.argv) == 4:
        _,model, type_supervised, unlabel_percent = sys.argv
        type_name = 'all'
        net_type = 'densenet121' 
        dataset, dataset_anomaly = 'svhn', 'cifar10'
        if model == 'ovae':
            lambda_ae, lambda_ce, lambda_mem, lambda_ce_sigmoid = '1','0','0','1'
        elif model=='baseline':
            lambda_ae, lambda_ce, lambda_mem, lambda_ce_sigmoid = '0','1','0','0'

    print('read_res get lambdas\n',dataset, dataset_anomaly, type_name, net_type, lambda_ae, lambda_ce, lambda_mem, lambda_ce_sigmoid, type_supervised, unlabel_percent)
    data = readres(dataset, dataset_anomaly, type_name, net_type, lambda_ae, lambda_ce, lambda_mem, lambda_ce_sigmoid, type_supervised, unlabel_percent)
    print(''.join(data))


if __name__ == '__main__':
    main()
