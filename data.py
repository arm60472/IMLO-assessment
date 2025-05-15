import torch
from torchvision import datasets, transforms

def get_dataloaders(batch_size=128, num_workers=4):
    # Training transforms (with augmentation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616)),
    ])
    
    # Validation/test transforms (no augmentation)
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616)),
    ])

    # Load datasets
    train_set = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform_val_test)

    # Split train into train+val
    num_train = len(train_set)
    val_size = int(0.1 * num_train)
    train_size = num_train - val_size
    
    # Use same transform for both subsets
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [train_size, val_size])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers)
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers)

    return train_loader, val_loader, test_loader