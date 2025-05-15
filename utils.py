import torch

def save_checkpoint(state_dict, filename):
    torch.save(state_dict, filename)

def load_checkpoint(filename):
    return torch.load(filename, map_location='cpu')

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count