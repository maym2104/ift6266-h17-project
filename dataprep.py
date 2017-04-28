#Modified from: https://github.com/sherjilozair/ift6266
#Credit:        Sherjil Ozair

import imageio
import glob
import numpy as np
import sys,os
from tqdm import tqdm

home = '.\\'
train_path = 'coco\\images\\train2014'
valid_path = 'coco\\images\\val2014'

images = []

for fname in tqdm(glob.glob('{}\\*.jpg'.format(home+train_path))):
    img = imageio.imread(fname)
    if img.shape == (64, 64, 3) and img.dtype == np.uint8:
        caption_id = os.path.basename(fname)[:-4]
        images.append((img,caption_id))
        #images[caption_id] = img

images = np.array(images, dtype=[('values', 'u1', (64, 64, 3)),('keys', 'S27')])
np.savez_compressed(home+'images.train.npz', images)

images = []

for fname in tqdm(glob.glob('{}\\*.jpg'.format(home+valid_path))):
    img = imageio.imread(fname)
    if img.shape == (64, 64, 3) and img.dtype == np.uint8:
        caption_id = os.path.basename(fname)[:-4]
        images.append((img,caption_id))
        #images[caption_id] = img

images = np.array(images, dtype=[('values', 'u1', (64, 64, 3)),('keys', 'S25')])
np.savez_compressed(home+'images.valid.npz', images)
