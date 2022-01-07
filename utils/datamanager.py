import cv2 as cv
import numpy as np
from utils.config import *
import os
from torch.utils.data import Dataset, DataLoader

def get_dirs(dir):
    X_files = os.listdir(dir+'/X')
    Y_files = os.listdir(dir + '/Y')
    X, Y = [], []
    for file in X_files:
        X.append(os.path.join(dir+'/X/', file))
    for file in Y_files:
        Y.append(os.path.join(dir+'/Y/', file))
    return X, Y

def load_data(dir, n_class=21, height=320, width=480):
    train_files, label_files = get_dirs(dir)
    assert len(train_files) == len(label_files)

    num_samples = len(train_files)

    x = np.empty((num_samples, 3, height, width), dtype=np.float32)
    y = np.empty((num_samples, n_class, height, width), dtype=np.float32)

    for i, filename in enumerate(train_files):
        #x[i] = np.load(f'{dir}/X_npy/{i}.npy')
        #if f'{dir}/X_npy/{i}.npy' not in os.listdir():
        bgr_img = cv.imread(filename)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        x[i] = rgb_img / 255.
        #    np.save(f'{dir}/X_npy/{i}.npy', x[i])
        #else:
        #    x[i] = np.load(f'{dir}/X_npy/{i}.npy')

        #if i % 100 == 0:
        #    print(f'Input data : {i}/{len(train_files)}')
    print(f'Input data : {i+1}/{len(train_files)}\n')

    for i, filename in enumerate(label_files):
        # y[i] = np.load(f'{dir}/Y_npy/{i}.npy')
        #if f'{dir}/Y_npy/{i}.npy' not in os.listdir():
        bgr_img = cv.imread(filename.strip())
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        segmentation_mask = np.zeros((height, width, n_class), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(rgb_img == label, axis=-1).astype(float)
        segmentation_mask = np.transpose(segmentation_mask, (2, 0, 1))
        y[i] = segmentation_mask
        #    np.save(f'{dir}/Y_npy/{i}.npy', y[i])
        #else:
        #    y[i] = np.load(f'{dir}/Y_npy/{i}.npy')
        #if i % 100 == 0:
        #    print(f'Label data : {i}/{len(label_files)}')
    print(f'Label data : {i+1}/{len(label_files)}\n')

    return x, y

class PASCALVOCDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.x, self.y = load_data(dir)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)















