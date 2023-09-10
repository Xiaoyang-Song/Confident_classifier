##############################################
# This code is based on samples from pytorch #
##############################################
# Writer: Kimin Lee
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
from data_loader import DSET
import numpy as np
import torchvision.utils as vutils
import models
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(
    description='Training code - joint confidence')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset', default='CIFAR10-SVHN', help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--outf', default='.',
                    help='folder to output images and model checkpoints')

parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1,
                    help='learning rate decay')
parser.add_argument('--decreasing_lr', default='60',
                    help='decreasing strategy')
parser.add_argument('--num_classes', type=int,
                    default=10, help='the # of classes')
parser.add_argument('--beta', type=float, default=1,
                    help='penalty parameter for KL term')
parser.add_argument('--num_channels', type=int,
                    default=3, help='# of channels')

args = parser.parse_args()

# if args.dataset == 'CIFAR10-SVHN':
#     args.beta = 0.1
#     args.batch_size = 64

print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('load InD data for Experiment: ', args.dataset)
# train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize, args.dataroot)
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


print('Load model')
# model = models.vgg13()
model = models.densenet.DenseNet3(
    depth=100, num_classes=args.num_classes, input_channel=args.num_channels)
print(model)

print('load GAN')
nz = 100
netG = models.Generator(1, nz, 64, args.num_channels)  # ngpu, nz, ngf, nc
netD = models.Discriminator(1, args.num_channels, 64)  # ngpu, nc, ndf
# Initial setup for GAN
real_label = 1
fake_label = 0
criterion = nn.BCELoss()
fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)

if args.cuda:
    model.cuda()
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    fixed_noise = fixed_noise.cuda()
fixed_noise = Variable(fixed_noise)

print('Setup optimizer')
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=1e-4)


def adjust_opt(optAlg, optimizer, epoch, max_epoch):
    if optAlg == 'sgd':
        if epoch < max_epoch * 0.5:
            lr = 1e-1
        elif epoch == max_epoch * 0.5:
            lr = 1e-2
        elif epoch == max_epoch * 0.75:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))


def train(epoch):
    model.train()
    # for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    for batch_idx, (data, target) in enumerate(train_loader):

        gan_target = torch.FloatTensor(target.size()).fill_(0)
        uniform_dist = torch.Tensor(
            data.size(0), args.num_classes).fill_((1./args.num_classes))

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            gan_target, uniform_dist = gan_target.cuda(), uniform_dist.cuda()

        data, target, uniform_dist = Variable(
            data), Variable(target), Variable(uniform_dist)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        # print(output.shape)
        # print(targetv.shape)
        targetv = targetv.unsqueeze(-1)
        errD_real = criterion(output, targetv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        if args.cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        targetv = Variable(gan_target.fill_(fake_label))
        targetv = targetv.unsqueeze(-1)
        output = netD(fake.detach())
        errD_fake = criterion(output, targetv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        targetv = targetv.unsqueeze(-1)
        errG = criterion(output, targetv)
        D_G_z2 = output.data.mean()

        # minimize the true distribution
        KL_fake_output = F.log_softmax(model(fake), dim=-1)
        errG_KL = F.kl_div(KL_fake_output, uniform_dist)*args.num_classes
        generator_loss = errG + args.beta*errG_KL
        generator_loss.backward()
        optimizerG.step()

        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=-1)
        loss = F.nll_loss(output, target)

        # KL divergence
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        if args.cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        KL_fake_output = F.log_softmax(model(fake), dim=-1)
        KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist)*args.num_classes
        total_loss = loss + args.beta*KL_loss_fake
        total_loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))
            fake = netG(fixed_noise)
            vutils.save_image(fake.data, '%s/gan_samples_epoch_%03d.png' %
                              (args.outf, epoch), normalize=True)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            total += data.size(0)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = F.log_softmax(model(data), dim=-1)
            test_loss += F.nll_loss(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        # loss function already averages over batch size
        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, total,
            100. * correct / total))


for epoch in tqdm(range(1, args.epochs + 1)):
    adjust_opt('sgd', optimizer, epoch, args.epochs)
    train(epoch)
    test(epoch)
    if epoch in decreasing_lr:
        optimizerG.param_groups[0]['lr'] *= args.droprate
        optimizerD.param_groups[0]['lr'] *= args.droprate
        # optimizer.param_groups[0]['lr'] *= args.droprate
    if epoch % 20 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' %
                   (args.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' %
                   (args.outf, epoch))
        torch.save(model.state_dict(), '%s/model_epoch_%d.pth' %
                   (args.outf, epoch))
