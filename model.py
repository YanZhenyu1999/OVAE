import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models import *

class gelu(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Auto_Encoder_MNIST(nn.Module):
    def __init__(self, args):
        super(Auto_Encoder_MNIST, self).__init__()
        self.fc_encoder1 = nn.Linear(28*28, 1200)
        self.fc_encoder2 = nn.Linear(1200, 1200)
        self.fc_encoder3 = nn.Linear(1200, 1200)
        self.fc_encoder4 = nn.Linear(1200, args.z_dim*2)

        self.fc_decoder1 = nn.Linear(args.z_dim, 1200)
        self.fc_decoder2 = nn.Linear(1200, 1200)
        self.fc_decoder3 = nn.Linear(1200, 1200)
        self.fc_decoder4 = nn.Linear(1200, 28*28)

        self.act = gelu()
    
    def encode(self, x):
        x = x.view(-1, 28*28)
        x = self.act(self.fc_encoder1(x))
        x = self.act(self.fc_encoder2(x))
        x = self.act(self.fc_encoder3(x))
        x = self.fc_encoder4(x)
        return x
    def decode(self, x):
        x = self.act(self.fc_decoder1(x))
        x = self.act(self.fc_decoder2(x))
        x = self.act(self.fc_decoder3(x))
        x = self.fc_decoder4(x)
        x = x.view(-1,1,28,28)
        return x

class Multi_Classifier_MNIST(nn.Module):
    def __init__(self, args):
        super(Multi_Classifier_MNIST, self).__init__()
        self.fc2 = nn.Linear(args.z_dim, 10)
        self.act = gelu()

    def forward(self, x):
        x = self.act(x)
        x = self.fc2(x)
        return x


class Auto_Encoder_CIFAR(nn.Module):
    def __init__(self, args):
        super(Auto_Encoder_CIFAR, self).__init__()
        # self.conv_encoder1 = nn.Conv2d(3, 32, 5, 2)
        # self.conv_encoder2 = nn.Conv2d(32, 64, 5, 2)
        # self.fc_encoder1 = nn.Linear(5*5*64, 512)
        # self.fc_encoder2 = nn.Linear(512, args.z_dim*2)
        # resnet for cifar10
        if args.net_type == 'resnet18':
            self.encode_net = ResNet18(args.z_dim*2)
        elif args.net_type == 'resnet34':
            self.encode_net = ResNet34(args.z_dim*2)
        elif args.net_type == 'densenet121':
            self.encode_net = DenseNet121(args.z_dim*2)
        elif args.net_type == 'densenet169':
            self.encode_net = DenseNet169(args.z_dim*2)
        elif args.net_type == 'densenet201':
            self.encode_net = DenseNet201(args.z_dim*2)


        # self.fc_decoder1 = nn.Linear(args.z_dim, 512)
        # self.fc_decoder2 = nn.Linear(512, 5*5*64)
        # self.conv_decoder1 = nn.ConvTranspose2d(64, 32, 6, 2)
        # self.conv_decoder2 = nn.ConvTranspose2d(32, 1, 6, 2)
        # self.fc_decoder3 = nn.Linear(5*5*64, 1200)
        # self.fc_decoder4 = nn.Linear(1200, 32*32*3)

        # de_conv
        self.fc_decoder1 = nn.Linear(args.z_dim, 256)
        self.conv_decoder1 = nn.ConvTranspose2d(256, 64, 4)      # 64, 4, 4
        self.conv_decoder2 = nn.ConvTranspose2d(64, 64, 4, 2, 1) # 64, 8, 8
        self.conv_decoder3 = nn.ConvTranspose2d(64, 32, 4, 2, 1) # 32, 16, 16
        self.conv_decoder4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)  # 3, 32, 32




        self.act = gelu()
    
    def encode(self, x):
        x = self.encode_net(x)
        return x

    def decode(self, x):

        # x = self.act(self.fc_decoder1(x))
        # x = self.act(self.fc_decoder2(x))
        # x = self.act(self.fc_decoder3(x))
        # x = self.fc_decoder4(x)
        # x = x.view(-1,3,32,32)

        # de_conv
        x = self.act(self.fc_decoder1(x))
        x = x.view(-1,256,1,1)
        x = self.act(self.conv_decoder1(x))
        x = self.act(self.conv_decoder2(x))
        x = self.act(self.conv_decoder3(x))
        x = self.conv_decoder4(x)
        return x

class Multi_Classifier_CIFAR(nn.Module):
    def __init__(self, args):
        super(Multi_Classifier_CIFAR, self).__init__()
        # self.fc1 = nn.Linear(args.z_dim, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 10)
        self.fc1 = nn.Linear(args.z_dim, 10)
        self.act = gelu()

    def forward(self, x):
        x = self.act(x)
        # x = self.act(self.fc1(x))
        # x = self.act(self.fc2(x))
        # x = self.fc3(x)
        x = self.fc1(x)
        return x


class Auto_Encoder_CIFAR100(nn.Module):
    def __init__(self, args):
        super(Auto_Encoder_CIFAR100, self).__init__()
        self.fc_encoder1 = nn.Linear(3*32*32, 4096)
        self.fc_encoder2 = nn.Linear(4096, 512)
        self.fc_encoder3 = nn.Linear(512, args.z_dim*2)

        if args.net_type == 'resnet18':
            self.encode_net = ResNet18(args.z_dim*2)
        elif args.net_type == 'resnet34':
            self.encode_net = ResNet34(args.z_dim*2)
        elif args.net_type == 'densenet121':
            self.encode_net = DenseNet121(args.z_dim*2)
        elif args.net_type == 'densenet169':
            self.encode_net = DenseNet169(args.z_dim*2)
        elif args.net_type == 'densenet201':
            self.encode_net = DenseNet201(args.z_dim*2)


        # self.fc_decoder1 = nn.Linear(args.z_dim, 512)
        # self.fc_decoder2 = nn.Linear(512, 4096)
        # self.fc_decoder3 = nn.Linear(4096, 32*32*3)

        # deconv
        self.fc_decoder1 = nn.Linear(args.z_dim, 256)
        self.conv_decoder1 = nn.ConvTranspose2d(256, 64, 4)      # 64, 4, 4
        self.conv_decoder2 = nn.ConvTranspose2d(64, 64, 4, 2, 1) # 64, 8, 8
        self.conv_decoder3 = nn.ConvTranspose2d(64, 32, 4, 2, 1) # 32, 16, 16
        self.conv_decoder4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)  # 3, 32, 32

        self.act = gelu()
    
    def encode(self, x):
        x = self.encode_net(x)
        # x = x.view(-1, 32*32)
        # x = self.act(self.fc_encoder1(x))
        # x = self.act(self.fc_encoder2(x))
        # x = self.fc_encoder3(x)
        return x
    def decode(self, x):

        # x = self.act(self.fc_decoder1(x))
        # x = self.act(self.fc_decoder2(x))
        # x = self.fc_decoder3(x)
        # x = x.view(-1,3,32,32)
        
        # deconv
        x = self.act(self.fc_decoder1(x))
        x = x.view(-1,256,1,1)
        x = self.act(self.conv_decoder1(x))
        x = self.act(self.conv_decoder2(x))
        x = self.act(self.conv_decoder3(x))
        x = self.conv_decoder4(x)
        return x

class Multi_Classifier_CIFAR100(nn.Module):
    def __init__(self, args):
        super(Multi_Classifier_CIFAR100, self).__init__()
        # self.fc1 = nn.Linear(20, 16)
        self.fc2 = nn.Linear(args.z_dim, 100)
        self.act = gelu()

    def forward(self, x):
        x = self.act(x)
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Auto_Encoder_SVHN(nn.Module):
    def __init__(self, args):
        super(Auto_Encoder_SVHN, self).__init__()
        if args.net_type == 'resnet18':
            self.encode_net = ResNet18(args.z_dim*2)
        elif args.net_type == 'resnet34':
            self.encode_net = ResNet34(args.z_dim*2)
        elif args.net_type == 'densenet121':
            self.encode_net = DenseNet121(args.z_dim*2)
        elif args.net_type == 'densenet169':
            self.encode_net = DenseNet169(args.z_dim*2)
        elif args.net_type == 'densenet201':
            self.encode_net = DenseNet201(args.z_dim*2)

        self.fc_decoder1 = nn.Linear(args.z_dim, 512)
        self.fc_decoder2 = nn.Linear(512, 5*5*64)
        self.conv_decoder1 = nn.ConvTranspose2d(64, 32, 6, 2)
        self.conv_decoder2 = nn.ConvTranspose2d(32, 1, 6, 2)
        self.fc_decoder3 = nn.Linear(5*5*64, 1200)
        self.fc_decoder4 = nn.Linear(1200, 32*32*3)

        self.act = gelu()
    
    def encode(self, x):
        x = self.encode_net(x)
        return x

    def decode(self, x):
        x = self.act(self.fc_decoder1(x))
        x = self.act(self.fc_decoder2(x))
        x = self.act(self.fc_decoder3(x))
        x = self.fc_decoder4(x)
        x = x.view(-1,3,32,32)
        return x

class Multi_Classifier_SVHN(nn.Module):
    def __init__(self, args):
        super(Multi_Classifier_SVHN, self).__init__()
        self.fc1 = nn.Linear(args.z_dim, 10)
        self.act = gelu()

    def forward(self, x):
        x = self.act(x)
        x = self.fc1(x)
        return x





