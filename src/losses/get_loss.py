import torch


def get_loss(loss_name: str = 'L1'):
    if loss_name == 'L1':
        return torch.nn.L1Loss()
