import random
import shutil
import time
import torch
# from torch.utils.tensorboard import SummaryWriter

from utils.visualization import *
from loguru import logger

def get_optimizer_from_args(model, lr, weight_decay, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                             weight_decay=weight_decay)


def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reset_gpu_peak_memory(device):
    """Start a new CUDA peak-memory measurement window."""
    if torch.cuda.is_available() and torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak_memory_mib(device):
    """Return allocator peaks for the current measurement window in MiB."""
    if not (torch.cuda.is_available() and torch.device(device).type == 'cuda'):
        return {'allocated': 0.0, 'reserved': 0.0}
    torch.cuda.synchronize(device)
    return {
        'allocated': torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        'reserved': torch.cuda.max_memory_reserved(device) / (1024.0 ** 2),
    }


def update_gpu_peak_memory(current, observed):
    return {
        'allocated': max(current['allocated'], observed['allocated']),
        'reserved': max(current['reserved'], observed['reserved']),
    }


def get_dir_from_args(TASK, root_dir, csv_path=None, run_name='', **kwargs):

    k_shot = kwargs['k_shot']
    dataset = kwargs['dataset']

    run_dir = os.path.join(root_dir, f'{dataset}', f'k_{k_shot}', run_name) if run_name else \
        os.path.join(root_dir, f'{dataset}', f'k_{k_shot}')
    csv_dir = os.path.join(run_dir, 'csv')
    check_dir = os.path.join(run_dir, 'checkpoint')
    img_dir = os.path.join(run_dir, 'imgs')

    csv_path = csv_path or os.path.join(csv_dir, f"Seed_{kwargs['seed']}-results.csv")
    check_path = os.path.join(check_dir, f"{TASK}-Seed_{kwargs['seed']}-{kwargs['class_name']}-check_point.pt")

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    return img_dir, csv_path, check_path
