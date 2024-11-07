import os
import logging
import argparse
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import CIFAR10
from torchmetrics import Accuracy

from models.convnet import ConvNet
from datasets import CIFAR10_MEAN, CIFAR10_STD
from engines import train_one_epoch, eval_one_epoch
from utils import save_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', default='convnet_v2', type=str)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str)

    # data
    parser.add_argument('--data', default='data', type=str)
    parser.add_argument('--batch_size', default=256, type=int)

    # model
    parser.add_argument('--blocks', nargs='+', default=[2, 2, 6, 2], type=int)
    parser.add_argument('--dims', nargs='+', default=[64, 128, 256, 512], type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--droppath', default=0.2, type=float)

    # train
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--mixup_alpha', default=0.8, type=float)
    parser.add_argument('--cutmix_alpha', default=1.0, type=float)    
    parser.add_argument('--weight_decays', default=0.05, type=float)
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--random_erasing', default=0.25, type=float)
    parser.add_argument('--random_crop', default=4, type=int)
    parser.add_argument('--random_horizontal_flip', default=0.5, type=float)
    parser.add_argument('--random_augment', nargs='+', default=(2, 9), type=int)
    
    args = parser.parse_args()
    return args


def main(args):
    # -------------------------------------------------------------------------
    # Set Logger & Checkpoint Dirs
    # -------------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.basicConfig(
        filename=f'{args.title}.log',
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
    )
    
    # -------------------------------------------------------------------------
    # Data Processing Pipeline
    # -------------------------------------------------------------------------
    train_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=args.random_crop),
        v2.RandomHorizontalFlip(p=args.random_horizontal_flip),
        v2.RandAugment(num_ops=args.random_augment[0], magnitude=args.random_augment[1]),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomErasing(p=args.random_erasing),
        v2.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    cutmix = v2.CutMix(num_classes=args.num_classes)
    mixup = v2.MixUp(num_classes=args.num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    train_data = CIFAR10(
        root=args.data, 
        train=True, 
        download=True, 
        transform=train_transform,
    )
    train_loader = DataLoader(
        dataset=train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=3, 
        collate_fn=collate_fn, 
        drop_last=True,
    )

    val_transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    val_data = CIFAR10(
        root=args.data, 
        train=False, 
        download=True, 
        transform=val_transform,
    )
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = ConvNet(
        blocks=args.blocks, 
        dims=args.dims, 
        kernel_size=args.kernel_size,
        droppath=args.droppath, 
        dropout=args.dropout, 
        num_classes=args.num_classes,
    )
    model = model.to(args.device)


    # -------------------------------------------------------------------------
    # Performance Metic, Loss Function, Optimizer
    # -------------------------------------------------------------------------
    metric_fn = Accuracy(task='multiclass', num_classes=args.num_classes)
    metric_fn = metric_fn.to(args.device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), args.lr, weight_decay=args.weight_decays)
    scheduler = CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

    # -------------------------------------------------------------------------
    # Run Main Loop
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        # train one epoch
        train_summary = train_one_epoch(
            model=model, 
            loader=train_loader, 
            loss_fn=loss_fn,
            optimizer=optimizer, 
            scheduler=scheduler, 
            device=args.device,
        )

        # evaluate one epoch
        val_summary = eval_one_epoch(
            model=model, 
            loader=val_loader, 
            metric_fn=metric_fn, 
            loss_fn=loss_fn, 
            device=args.device,
        )

        # write log
        log = (f'epoch {epoch+1}, '
               + f'train_loss: {train_summary["loss"]:.4f}, '
               + f'val_loss: {val_summary["loss"]:.4f}, '
               + f'val_accuracy: {val_summary["accuracy"]:.4f}')
        
        print(log)
        logging.info(log)

        # save model
        checkpoint_path = f'{args.checkpoint_dir}/{args.title}_last.pt'
        save_model(checkpoint_path, model, optimizer, scheduler, epoch+1)


if __name__=="__main__":
    args = get_args()
    main(args)