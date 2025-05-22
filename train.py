import torch
import torch.optim as optim
import torch.nn as nn
from data import get_dataloaders
from model import CIFAR10Net
from utils import save_checkpoint, AverageMeter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm  # Added for progress bar
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import argparse

def train(args):
    device = torch.device('cpu')
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    net = CIFAR10Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        net.train()
        losses = AverageMeter()
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        train_acc = 100. * correct / total
        val_acc = validate(net, val_loader, criterion, device)
        scheduler.step()

        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train Loss: {losses.avg:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Acc: {val_acc:.2f}%')

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(net.state_dict(), 'best_model.pth')
    print(f'Best Val Acc: {best_val_acc:.2f}%')


def validate(net, loader, criterion, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

if __name__ == '__main__':
    class Args:
        batch_size = 128
        epochs = 100
        lr = 0.1
        workers = 2
    
    train(Args())