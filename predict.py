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

    print(f"Device: {device}")
    print(
        f"Infering for {arg.type} speaker recognition | "
        f"Total number of speakers: {arg.spks} | "
        f"Number of speakers per batch: {arg.batch_size} | "
        f"Model selected: {arg.model} | "
        f"Pre-processed model selected: {arg.pre_model} | "
        f"Data augmentation: {arg.data_augmentation} | "
    )

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

    _, test_loader, _ = dataloader(arg)

    if not os.path.exists(arg.ckpt):
        raise Exception("No checkpoint directory found")

    model_config = UAIConfig(arg)
    model = UAI(model_config, arg)
    if arg.data_augmentation:
        ckpt_path = f"{arg.ckpt}/{arg.type}/epoch_{arg.type}_{arg.pre_model}_{arg.evaluation}_aug.pth"
    else:
        ckpt_path = f"{arg.ckpt}/{arg.type}/epoch_{arg.type}_{arg.pre_model}_{arg.evaluation}.pth"
    best_model_ckpt = torch.load(
        ckpt_path,
        map_location = device
    )
    model.init_model()
    model.load_state_dict(best_model_ckpt['model_state_dict'])

    infer = Inference(arg, device, model)
    test_acc = infer.evaluate(test_loader)

    print(f"Test accuracy: {test_acc:.4f}")

    print('-'*50)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type = str, default = 'config.yaml',
                        help = 'arg file path')
    parser.add_argument('--gpu', type = str, default = '0', help = 'gpu id')
    args = parser.parse_args()
    main(args.cfg, args.gpu)