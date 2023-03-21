import sys
import subprocess
import pdb

model = 'ovae'
python_version = 3
run_state = 'train'

## sudo python3 run.py 0 train svhn cifar10 1 0 0 1 densenet121 semi 0.0
## sudo python3 run.py 0 test_normal svhn cifar10 1 0 0 1 densenet121 semi 0.0
## sudo python3 run.py 0 test svhn cifar10 1 0 0 1 densenet121 semi 0.0

#pdb.set_trace()
if len(sys.argv) == 12:
    _, python_version, run_state, dataset, anomaly_dataset, ae, ce, mem, ce_sigmoid, net_type, type_supervised, unlabel_percent = sys.argv
    print('run receive the lambdas\n',python_version, run_state, dataset, anomaly_dataset, ae, ce, mem, ce_sigmoid, net_type, type_supervised, unlabel_percent)
elif len(sys.argv) == 6:
    _, python_version, run_state, model, type_supervised, unlabel_percent = sys.argv
    print('run receive the lambdas\n', python_version, run_state, model, type_supervised, unlabel_percent)

# default
dataset = 'svhn'
anomaly_dataset = 'cifar10'
net_type = 'densenet121'

if model == 'ovae':
    ae, ce, mem, ce_sigmoid = '1', '0', '0', '1'
elif model == 'baseline':
    ae, ce, mem, ce_sigmoid = '0', '1', '0', '0'
print('simple input => ', python_version, run_state, model, type_supervised, unlabel_percent)


print('the lambdas =',python_version, run_state, dataset, anomaly_dataset, ae, ce, mem, ce_sigmoid, net_type, type_supervised, unlabel_percent)

cmd = 'CUDA_VISIBLE_DEVICES='+str(int(python_version))+' /home/lq/anaconda3/bin/python'
cmd += ' main.py'
cmd += ' --run-state '+str(run_state)
cmd += ' --dataset '+str(dataset)
cmd += ' --anomaly-dataset '+str(anomaly_dataset)
cmd += ' --lambda-ae '+str(ae)
cmd += ' --lambda-ce '+str(ce)
cmd += ' --lambda-mem '+str(mem)
cmd += ' --lambda-ce-sigmoid '+str(ce_sigmoid)
cmd += ' --net-type '+str(net_type)
cmd += ' --type-supervised '+str(type_supervised)
cmd += ' --unlabel-percent '+str(unlabel_percent)
subprocess.call(cmd, shell=True)

