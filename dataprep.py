#Credit: Sherjil Ozair ()

import imageio
import glob
import numpy as np
from tqdm import tqdm

home = 'C:\\Users\\Mariane\\Documents\\ift6266\\ift6266-h17-project\\'
path = 'coco\\images\\train2014'

images = []

for fname in tqdm(glob.glob('{}\\*.jpg'.format(home+path))):
    img = imageio.imread(fname)
    if img.shape == (64, 64, 3) and img.dtype == np.uint8:
        images.append(img)

images = np.array(images)
np.savez_compressed(home+'images.train.npz', images)
