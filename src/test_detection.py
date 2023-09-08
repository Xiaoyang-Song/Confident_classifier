###################################################################################################
# Measure the detection performance: reference code is https://github.com/ShiyuLiang/odin-pytorch #
###################################################################################################
# Writer: Kimin Lee
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import torchvision.utils as vutils
import calculate_log as callog
import models
import math
from data_loader import *
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from numpy.linalg import inv

# Training settings
parser = argparse.ArgumentParser(
    description='Test code - measure the detection peformance')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--dataset', required=True,
                    help='target dataset: cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--outf', default='/home/rack/KM/2017_Codes/overconfidence/test/log_entropy',
                    help='folder to output images and model checkpoints')
parser.add_argument('--out_dataset', required=True,
                    help='out-of-dist dataset: cifar10 | svhn | imagenet | lsun')
parser.add_argument('--num_classes', type=int, default=10,
                    help='number of classes (default: 10)')
parser.add_argument('--num_channels', type=int,
                    default=3, help='# of channels')
parser.add_argument('--pre_trained_net', default='',
                    help="path to pre trained_net")

args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('Load model')
# model = models.vgg13()
model = models.densenet.DenseNet3(
    depth=100, num_classes=args.num_classes, input_channel=args.num_channels)
model.load_state_dict(torch.load(args.pre_trained_net))
print(model)

print('load target data: ', args.dataset)
if args.dataset == 'CIFAR10-SVHN' or args.dataset == 'MNIST-FashionMNIST':
    dset = DSET(args.dataset, False, args.batch_size,
                args.batch_size, None, None)
    train_loader, test_loader = dset.ind_train_loader, dset.ind_val_loader
elif args.dataset == 'SVHN' or args.dataset == 'FashionMNIST':
    dset = DSET(args.dataset, True, args.batch_size,
                args.batch_size, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
    train_loader, test_loader = dset.ind_train_loader, dset.ind_val_loader
elif args.dataset == 'MNIST':
    dset = DSET(args.dataset, True, args.batch_size,
                args.batch_size, [2, 3, 6, 8, 9], [1, 7])
    train_loader, test_loader = dset.ind_train_loader, dset.ind_val_loader
else:
    assert False

print('load non target data: ', args.out_dataset)

if args.dataset == 'CIFAR10-SVHN' or args.dataset == 'MNIST-FashionMNIST':
    dset = DSET(args.dataset, False, args.batch_size,
                args.batch_size, None, None)
    nt_test_loader = dset.ood_val_loader
elif args.dataset == 'SVHN' or args.dataset == 'FashionMNIST':
    dset = DSET(args.dataset, True, args.batch_size,
                args.batch_size, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
    nt_test_loader = dset.ood_val_loader
elif args.dataset == 'MNIST':
    dset = DSET(args.dataset, True, args.batch_size,
                args.batch_size, [2, 3, 6, 8, 9], [1, 7])
    nt_test_loader = dset.ood_val_loader
else:
    assert False

if args.cuda:
    model.cuda()


def generate_target():
    model.eval()
    correct = 0
    total = 0
    f1 = open('%s/confidence_Base_In.txt' % args.outf, 'w')
    with torch.no_grad():
        for data, target in test_loader:
            total += data.size(0)
            # vutils.save_image(data, '%s/target_samples.png'%args.outf, normalize=True)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            batch_output = model(data)

            # compute the accuracy
            pred = batch_output.data.max(1)[1]
            equal_flag = pred.eq(target.data).cpu()
            correct += equal_flag.sum()
            for i in range(data.size(0)):
                # confidence score: max_y p(y|x)
                output = batch_output[i].view(1, -1)
                soft_out = F.softmax(output, dim=-1)
                soft_out = torch.max(soft_out.data)
                f1.write("{}\n".format(soft_out))

        print('\n Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct,
                                                            total, 100. * correct / total))


def generate_non_target():
    model.eval()
    total = 0
    f2 = open('%s/confidence_Base_Out.txt' % args.outf, 'w')
    with torch.no_grad():
        for data, target in nt_test_loader:
            total += data.size(0)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            batch_output = model(data)
            for i in range(data.size(0)):
                # confidence score: max_y p(y|x)
                output = batch_output[i].view(1, -1)
                soft_out = F.softmax(output, dim=-1)
                soft_out = torch.max(soft_out.data)
                f2.write("{}\n".format(soft_out))


print('generate log from in-distribution data')
generate_target()
print('generate log  from out-of-distribution data')
generate_non_target()
print('calculate metrics')
callog.metric(args.outf)
