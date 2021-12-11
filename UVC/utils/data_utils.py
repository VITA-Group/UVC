import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import os

logger = logging.getLogger(__name__)



def get_loader(args):

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()


    if args.dataset == "cifar10" or args.dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        if args.dataset == "cifar10":

            trainset = datasets.CIFAR10(root=args.data_dir,
                                        train=True,
                                        download=True,
                                        transform=transform_train)
            testset = datasets.CIFAR10(root=args.data_dir,
                                       train=False,
                                       download=True,
                                       transform=transform_test) if args.local_rank in [-1, 0] else None
        else:
            trainset = datasets.CIFAR100(root="./data",
                                         train=True,
                                         download=True,
                                         transform=transform_train)
            testset = datasets.CIFAR100(root="./data",
                                        train=False,
                                        download=True,
                                        transform=transform_test) if args.local_rank in [-1, 0] else None

        if args.local_rank == 0:
            torch.distributed.barrier()

        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)

        test_sampler = SequentialSampler(testset)
        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=32,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=32,
                                 pin_memory=True) if testset is not None else None

    elif args.dataset == "imagenet":
        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.local_rank == 0:
            torch.distributed.barrier()


        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)




    return train_loader, test_loader
