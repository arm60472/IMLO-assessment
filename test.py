import torch
from data import get_dataloaders
from model import CIFAR10Net
from utils import load_checkpoint, AverageMeter


def test():
    device = torch.device('cpu')
    _, _, test_loader = get_dataloaders(batch_size=128, num_workers=4)
    net = CIFAR10Net().to(device)
    state = load_checkpoint('best_model.pth')
    net.load_state_dict(state)
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f'Test Accuracy: {100. * correct / total:.2f}%')

if __name__ == '__main__':
    test()