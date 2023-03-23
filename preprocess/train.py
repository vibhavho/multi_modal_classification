import os
import sys
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


from engine import *
from models.threedcnn import ThreeDCNN
from models.cv_lstm import ConvLSTMAutoEncoder


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset import VidTimit
from utils import Configuration, SaveBestModel, save_model, video_paths



def main(cfg, gpu):
    print('-'*50)

    assert isinstance(cfg, object)
    cfg = Configuration(cfg)

    if gpu is not None:
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    # Load data
    dataset = VidTimit(video_paths(cfg.data), cfg)

    loader = DataLoader(dataset,
                        batch_size = 1,
                        shuffle = True,
                        num_workers = cfg.num_workers,
                        pin_memory = True)

    if cfg.data_aug:
        print("Data Augmentation Enabled")
    else:
        print("Data Augmentation Disabled")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(cfg.ckpt):
        os.makedirs(cfg.ckpt)

    if cfg.model == '3dcnn': model = ThreeDCNN().to(device)
    elif cfg.model == 'convlstm': model = ConvLSTMAutoEncoder().to(device)
    else:
        raise Exception("Preprocessing model not found")

    optimiser = optim.Adam(model.parameters(), 
                          lr = cfg.lr,
                          weight_decay = 1e-8) if cfg.optim == 'adam' else \
                optim.SGD(model.parameters(), lr = cfg.lr, momentum = 0.9)

    loss = nn.CrossEntropyLoss() if cfg.loss == 'cross_entropy' else \
           nn.MSELoss()

    scheduler = StepLR(optimiser, step_size = 1, gamma = 0.5)

    save_best_model = SaveBestModel()
    
    for epoch in range(cfg.epochs):
        print(f"Epoch: {epoch}/{cfg.epochs}")
        epoch_loss = trainer(
            cfg,
            model,
            device, 
            loader,
            optimiser,
            loss,
            scheduler
        )

        save_best_model(
            epoch_loss,
            epoch, 
            model, 
            optimiser,
            loss,
            cfg.ckpt,
            cfg.data_aug,
            cfg.model
        )   
        print(f"Loss: {epoch_loss}")

    save_model(
        cfg.epochs,
        model,
        optimiser,
        loss, 
        cfg.ckpt,
        cfg.data_aug,
        cfg.model
    )

    print('TRAINING COMPLETE')
    print('-'*50)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Feature extraction')
    parser.add_argument('--config', default = '/ifs/loni/faculty/shi/spectrum/Student_2020/sarthak/EE_641/preprocess/config.yaml', 
                        type = str, help = 'config file')
    parser.add_argument('--gpu', default = '0', type = str, 
                        help = 'assign gpu device')    
    args = parser.parse_args()

    main(args.config, args.gpu)


