import os
import sys
import random
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


from models.threedcnn import ThreeDCNN
from models.cv_lstm import ConvLSTMAutoEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import *
from dataset import *


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

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    dataset = VidTimit(video_paths(cfg.data), cfg)
    paths = video_paths(cfg.data)
    
    if cfg.model == '3dcnn': model = ThreeDCNN().to(device)
    elif cfg.model == 'convlstm': model = ConvLSTMAutoEncoder().to(device)
    else:
        raise Exception("Preprocessing model not found")

    ckpt = torch.load(
        os.path.join(cfg.ckpt, cfg.model, 'best_model.pth'),
    )
    model.load_state_dict(ckpt['model'])
    model.eval()

    with torch.no_grad():
        for curr_idx in range(len(paths)):
            curr_path = paths[curr_idx]
            vid_feat = dataset.__getitem__(curr_idx).to(device)
            if cfg.model == 'convlstm':
                vid_feat = vid_feat.view(vid_feat.shape[0] * vid_feat.shape[2], 
                           vid_feat.shape[1], vid_feat.shape[3], vid_feat.shape[4])
            _, emb = model(vid_feat)
            if cfg.model == 'convlstm':
                emb = F.avg_pool2d(emb, emb.shape[2:]).view(emb.shape[0], -1)
            else:
                emb = F.avg_pool3d(emb, emb.shape[2:]).view(emb.shape[0], -1)
            # final_emb = torch.mean(emb, dim = 0).cpu().numpy()
            curr_folder = '/'.join(curr_path.split('/')[-3:-1])
            save_path = f"{cfg.processed}/{curr_folder}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if cfg.data_aug:
                save = f"{save_path}/{curr_path.split('/')[-1]}_{cfg.model}_aug.npy"
            else:
                save = f"{save_path}/{curr_path.split('/')[-1]}_{cfg.model}.npy"
            np.save(save, emb.cpu().numpy())
        
    print('Done!')
    print('-'*50)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = '/ifs/loni/faculty/shi/spectrum/Student_2020/sarthak/EE_641/preprocess/config.yaml', 
                        type = str, help = 'config file')
    parser.add_argument('--gpu', default = '0', type = str, 
                        help = 'assign gpu device')    
    args = parser.parse_args()
    main(args.config, args.gpu)