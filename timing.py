import sys
import subprocess


def run(python_version, dataset, anomaly_dataset, ae, ce, mem, ce_sigmoid, net_type):
    print(python_version, dataset, anomaly_dataset, ae, ce, mem, ce_sigmoid, net_type)
    cmd = 'CUDA_VISIBLE_DEVICES='+str(int(python_version))+' /home/lq/anaconda3/bin/python'
    cmd += ' main.py'
    cmd += ' --run-state train'
    cmd += ' --dataset '+str(dataset)
    cmd += ' --anomaly-dataset '+str(anomaly_dataset)
    cmd += ' --lambda-ae '+str(ae)
    cmd += ' --lambda-ce '+str(ce)
    cmd += ' --lambda-mem '+str(mem)
    cmd += ' --lambda-ce-sigmoid '+str(ce_sigmoid)
    cmd += ' --net-type '+str(net_type)
    cmd += ' --epochs '+str(int(1))
    subprocess.call(cmd, shell=True)

dataset = sys.argv[1]
python_version = int(sys.argv[2])
net_types = {'cifar':['resnet34', 'densenet121'],
            'cifar100':['resnet34', 'densenet121'],
            'svhn':['resnet34', 'densenet121'],
            'mnist':['mlp']}
ano = {'cifar':'gaussian',
        'cifar100':'gaussian',
        'svhn':'cifar10',
        'mnist':'gaussian'}


for net_type in  net_types[dataset]:
    run(python_version ,dataset, ano[dataset], 2, 0, 0, 0, net_type)
    # run(python_version ,dataset, ano[dataset], 1, 0, 0, 0, net_type)
    # run(python_version ,dataset, ano[dataset], 0, 1, 0, 0, net_type)
    # run(python_version ,dataset, ano[dataset], 1, 1, 0, 0, net_type)

