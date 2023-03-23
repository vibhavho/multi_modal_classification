import os
import cv2
import math
import numpy as np

import torch 
from torch.utils.data import Dataset


def resize_image(image, shape = (96, 96)):
    """
    Adopted from : https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv 
    """
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape)>2 else 1

    if h == w:
        return cv2.resize(image, shape, cv2.INTER_AREA)
    
    dif = h if h > w else w

    interp = cv2.INTER_AREA if dif > (int(float(shape[0])) + int(float(shape[1]))) / 2 else \
             cv2.INTER_CUBIC

    x_pos = int((dif - w) / 2.)
    y_pos = int((dif - h) / 2.)

    if len(image.shape) == 2:
        mask = np.zeros((dif, dif), dtype=image.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = image[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=image.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = image[:h, :w, :]

    return cv2.resize(mask, shape, interp)


class VidTimit(Dataset):
    def __init__(self, video_paths, cfg):
        self.h = 384
        self.w = 512
        self.c = 3
        self.cfg = cfg
        self.paths = video_paths
        self.crop_size = cfg.crop_size

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        frames = []
        cnt = 0
        fps = 25 # average fps of the dataset (given)
        for frame in os.listdir(self.paths[index]):
            frame_path = f"{self.paths[index]}/{frame}"
            frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            assert frame.shape == (self.h, self.w, self.c), \
            f"Frame shape is {frame.shape}. Expected shape is {(self.h, self.w, self.c)}"
            crop_frame = resize_image(frame, eval(self.crop_size))
            if self.cfg.data_aug:
                crop_frame = self.augment(crop_frame)
            frames += [crop_frame / 255.0]
            cnt += 1
        # print(f"Number of frames in {self.paths[index]} is {cnt}.")
        assert len(frames) != 0, "No frames found"
        overlap = math.ceil(fps * 0.5)
        frames = frames[: len(frames) - (len(frames) % overlap)]
        x = []
        for idx in range(0, len(frames), overlap):
            beg_idx, end_idx = idx, idx + int(fps)
            x.append(frames[beg_idx: end_idx])
        assert len(x) != 0, "No frames found"
        if len(x[-1]) != fps: x.pop()
        x = torch.from_numpy(np.array(x).astype(np.float32))
        return x.permute(0, 4, 1, 2, 3)


    def augment(self, frame, mean = 0, sigma = 0.1):
        """
        Add Gaussian noise to the frame, if data augmentation is enabled.
        """
        noisy_image = np.zeros(frame.shape, np.float32)
        noise = np.random.normal(mean, sigma, frame.shape)
        noisy_image[:,:,0] = frame[:,:,0] + noise[:,:,0]
        noisy_image[:,:,1] = frame[:,:,1] + noise[:,:,1]
        noisy_image[:,:,2] = frame[:,:,2] + noise[:,:,2]
        return noisy_image
