import os
import random
import warnings
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from src import *
from utils import * 
from models import *

warnings.filterwarnings("ignore")


def main(args, gpu):
    print('-'*50)

    assert isinstance(args, object)
    arg = Configuration(args)

    if gpu is not None:
        torch.manual_seed(arg.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(arg.seed)
        random.seed(arg.seed)

    device = torch.device(f"cuda:{gpu}" if \
             torch.cuda.is_available() else "cpu")

    main_worker(arg, device)

    print('-'*50)


def main_worker(arg, device):
    print(f"Device: {device}")
    print(
        f"Setting up for {arg.type} speaker recognition | "
        f"Total number of speakers: {arg.spks} | "
        f"Number of speakers per batch: {arg.batch_size} | "
        f"Model selected: {arg.model} | "
        f"Pre-processed model selected: {arg.pre_model} | "
        f"Data augmentation: {arg.data_augmentation} | "
    )

    print("Loading data...")
    train_loader, _, val_loader\
         = dataloader(arg)

    if not os.path.exists(arg.ckpt):
        os.makedirs(arg.ckpt)

    print("Initializing model...")
    model_config = UAIConfig(arg)
    model = UAI(model_config, arg)
    model.init_model()
    model.init_weights()

    print("Training...")
    t = Trainer(arg, device, model)
    epoch, val_acc = t.fit(train_loader, val_loader)

    print(
        f"Best epoch: {epoch},| "
        f"Best val acc: {val_acc}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type = str, default = 'config.yaml',
                        help = 'arg file path')
    parser.add_argument('--gpu', type = str, default = '1', help = 'gpu id')
    args = parser.parse_args()
    main(args.cfg, args.gpu)