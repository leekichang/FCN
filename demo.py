import torch
import os
from utils.config import *
import cv2 as cv
import numpy as np
from utils.image import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
height = 320
width = 480

checkpoint = './saved/FCN/BEST_checkpoint.tar'
print('checkpoint: ' + str(checkpoint))

# Load models
checkpoint = torch.load(checkpoint, map_location=torch.device(device))
model = checkpoint['model']
model = model.to(device)
model.eval()

test_path = './test_images/input'
test_images = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.png')]
test_images.sort()
num_test_samples = len(test_images)

imgs = torch.zeros([num_test_samples, 3, height, width], dtype=torch.float, device=device)

for i, path in enumerate(test_images):
    # Read images
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = img.transpose(2, 0, 1)
    assert img.shape == (3, height, width)
    assert np.max(img) <= 255
    img = torch.FloatTensor(img / 255.)
    imgs[i] = img

pred = model(imgs)
for idx, img in enumerate(pred[0]):
    get_image(img, idx)

#get_image(pred)