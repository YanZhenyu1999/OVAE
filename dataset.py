import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt


import pdb


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
        ])
transform_ano_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
        transforms.Pad(padding=2, fill=0, padding_mode='constant')],
        )
transform_to_tensor = transforms.Compose([transforms.ToTensor(),])


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def images(self, nums, channel, size):
        data = []
        for index in range(nums):
            path = self.imgs[index][0]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)[:channel,:size,:size]
            data.append(img)
        return data



def transform_gaussian_uniform(data):
    for i in range(data.shape[0]):
        data[i][0] = (data[i,0] - 125.3/255) / (63.0/255)
        data[i][1] = (data[i,1] - 123.0/255) / (62.1/255)
        data[i][2] = (data[i,2] - 113.9/255) / (66.7/255)
    return data


class dataset_myself():
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        d  = self.data[index] 
        l = self.label[index]
        return d,l

# normal dataset

class mnist_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size


    def getdata(self, args):
        train_dataset = datasets.MNIST(args.data_path, train=True, download=False, transform=transform_to_tensor)
        test_dataset = datasets.MNIST(args.data_path, train=False, download=False, transform=transform_to_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, **self.kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return train_loader, test_loader

class cifar_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
    def getdata(self, args):

        train_dataset = datasets.CIFAR10(args.data_path, train=True, download=False,transform=transform)
        test_dataset = datasets.CIFAR10(args.data_path, train=False, download=False,transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, **self.kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return train_loader, test_loader

class cifar100_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
    def getdata(self, args):
        train_dataset = datasets.CIFAR100(args.data_path, train=True, download=False,transform=transform)
        test_dataset = datasets.CIFAR100(args.data_path, train=False, download=False,transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, **self.kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return train_loader, test_loader

class svhn_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
    def getdata(self, args):
        train_dataset = datasets.SVHN(args.data_path, split='train', download=False,transform=transform)
        test_dataset = datasets.SVHN(args.data_path, split='test', download=False,transform=transform)
        
        #pdb.set_trace()
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size, shuffle=True, **self.kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return train_loader, test_loader


# anomaly dataset

class ano_gaussian_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.test_batch_size = args.test_batch_size
        self.data_num = args.anomaly_data_number
        if args.dataset == 'mnist':
            self.data_size = 28
            self.channel = 1
        elif args.dataset in ['cifar','cifar100','svhn']:
            self.data_size = 32
            self.channel = 3
        else:
            print('no dataset:',args.dataset)

    def getdata(self, args):
        data = np.random.normal(loc=0.0, scale=1.0, size=[self.data_num, self.channel, self.data_size, self.data_size]).astype(float)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        if args.dataset != 'mnist':
            data = transform_gaussian_uniform(data)
        
        label = np.zeros(self.data_num).astype(int)
        test_dataset = dataset_myself(data, label)
        # print('test anomaly: ',test_dataset.data.shape)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return test_loader


class ano_uniform_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.test_batch_size = args.test_batch_size
        self.data_num = args.anomaly_data_number
        if args.dataset == 'mnist':
            self.data_size = 28
            self.channel = 1
        elif args.dataset in ['cifar','cifar100','svhn']:
            self.data_size = 32
            self.channel = 3
        else:
            print('no dataset:',args.dataset)

    def getdata(self, args):
        data = np.random.uniform(low=0.0, high=1.0, size=[self.data_num, self.channel, self.data_size, self.data_size]).astype(float)
        if args.dataset != 'mnist':
            data = transform_gaussian_uniform(data)
        
        label = np.zeros(self.data_num).astype(int)
        test_dataset = dataset_myself(data, label)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return test_loader
        
class ano_cifar10_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.test_batch_size = args.test_batch_size
        if args.dataset == 'mnist':
            self.data_size = 28
            self.channel = 1
        elif args.dataset in ['cifar100', 'svhn']:
            self.data_size = 32
            self.channel = 3
        else:
            print('wrong dataset for cifar10:',args.dataset)
    def getdata(self, args):
        test_dataset = datasets.CIFAR10(args.ano_data_path, train=False, download=False,transform=transform)
        #pdb.set_trace()
        # test_dataset.data = test_dataset.data[:,:self.data_size,:self.data_size,:self.channel]
        test_dataset.test_labels = np.zeros(len(test_dataset.test_labels)).astype(int)
        # test_dataset.test_data = test_dataset.test_data[:,:self.data_size,:self.data_size,:self.channel]
        # test_dataset.test_labels = np.zeros(len(test_dataset.test_labels)).astype(int)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return test_loader


class ano_cifar100_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.test_batch_size = args.test_batch_size
        if args.dataset == 'mnist':
            self.data_size = 28
            self.channel = 1
        elif args.dataset in ['cifar','svhn']:
            self.data_size = 32
            self.channel = 3
        else:
            print('wrong dataset for cifar100:',args.dataset)
    def getdata(self, args):
        test_dataset = datasets.CIFAR100(args.ano_data_path, train=False, download=False,transform=transform)
        # test_dataset.data = test_dataset.data[:,:self.data_size,:self.data_size,:self.channel]
        test_dataset.targets = np.zeros(len(test_dataset.targets)).astype(int)
        # test_dataset.test_data = test_dataset.test_data[:,:self.data_size,:self.data_size,:self.channel]
        # test_dataset.test_labels = np.zeros(len(test_dataset.test_labels)).astype(int)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return test_loader

class ano_svhn_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.test_batch_size = args.test_batch_size
        self.data_num = args.anomaly_data_number
        if args.dataset == 'mnist':
            self.data_size = 28
            self.channel = 1
        elif args.dataset in ['cifar','cifar100']:
            self.data_size = 32
            self.channel = 3

        else:
            print('wrong dataset for svhn:',args.dataset)
    def getdata(self, args):
        test_dataset = datasets.SVHN(args.ano_data_path, split='test', download=False,transform=transform)
        test_dataset.data = test_dataset.data[:self.data_num,:self.channel,:self.data_size,:self.data_size]
        test_dataset.labels = np.zeros(self.data_num).astype(int)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return test_loader

class ano_lsun_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.test_batch_size = args.test_batch_size
        self.data_num = args.anomaly_data_number
        if args.dataset == 'mnist':
            self.data_size = 28
            self.channel = 1
        elif args.dataset in ['cifar','cifar100', 'svhn']:
            self.data_size = 32
            self.channel = 3
        else:
            print('wrong dataset for lsun:',args.dataset)
    def getdata(self, args):
        image_data = CustomImageFolder(args.ano_data_path, transform=transform)
        data = image_data.images(self.data_num, self.channel, self.data_size)
        label = np.zeros(self.data_num).astype(int)
        test_dataset = dataset_myself(data, label)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return test_loader

class ano_imagenet_dataset():
    def __init__(self, args):
        self.kwargs = {'num_workers': 1, 'pin_memory': True} 
        self.test_batch_size = args.test_batch_size
        self.data_num = args.anomaly_data_number
        if args.dataset == 'mnist':
            self.data_size = 28
            self.channel = 1
        elif args.dataset in ['cifar','cifar100','svhn']:
            self.data_size = 32
            self.channel = 3
        else:
            print('wrong dataset for imagenet:',args.dataset)
    def getdata(self, args):
        image_data = CustomImageFolder(args.ano_data_path, transform=transform)
        data = image_data.images(self.data_num, self.channel, self.data_size)
        label = np.zeros(self.data_num).astype(int)
        test_dataset = dataset_myself(data, label)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **self.kwargs)
        return test_loader










