import cv2 as cv
import numpy as np
import torch
from utils.config import *

height, width = 320, 480

def get_image(pred, k):
    classes_pred = torch.argmax(pred, dim=0)
    segmented_image = np.empty((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            segmented_image[i, j, :] = VOC_COLORMAP[classes_pred[i, j]]
    cv.imwrite(f'./{k}_out.png', segmented_image)
    print(f'Image saved at "./{k}_out.png"')