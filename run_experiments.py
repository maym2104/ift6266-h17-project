from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import models
import numpy as np
from settings import SAVED_DIR
import os

m = models.GenerativeAdversarialNetwork()
m.build()
m.load('model.pkl')
m.compile('generate_samples')

def to_image(img):
  np_img = ((img + 1.) * 127.5).astype('uint8').transpose(1, 2, 0)
  return Image.fromarray(np_img)

# font = ImageFont.truetype("sans-serif.ttf", 12)

bs = 100
z_dim = m.hparams.get('z_dim')
imgs, real, matching, fake, end = m.loader.get_next_batch(bs, 'test', skip_images=False, img_scale='tanh')
mask = np.zeros((bs, 3, 64, 64), dtype='float32')
mask[:, :, 16:48, 16:48] = 1.
mask_color = np.random.random_integers(0, 1, bs)[:, None, None, None] * mask
print('... Done creating mask')

cap_0 = m.create_one_hot_vector(real)
cap_100 = m.create_one_hot_vector(fake)
cap_12 = (0.125 * cap_100 + (1. - 0.125) * cap_0).astype('float32')
cap_25 = (0.250 * cap_100 + (1. - 0.250) * cap_0).astype('float32')
cap_37 = (0.375 * cap_100 + (1. - 0.375) * cap_0).astype('float32')
cap_50 = (0.500 * cap_100 + (1. - 0.500) * cap_0).astype('float32')
cap_62 = (0.625 * cap_100 + (1. - 0.625) * cap_0).astype('float32')
cap_75 = (0.750 * cap_100 + (1. - 0.750) * cap_0).astype('float32')
cap_87 = (0.875 * cap_100 + (1. - 0.875) * cap_0).astype('float32')
print ('... Done creating one hot vector')

# Generating images
cropped_images = (imgs * (1. - mask) + mask_color).astype('float32')
print('... Generating images ... ')
gen_1 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_0, 1)
print('... Generating images ... 1/15')
gen_2 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_0, 1)
print('... Generating images ... 2/15')
gen_3 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_0, 1)
print('... Generating images ... 3/15')
gen_4 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_0, 1)
print('... Generating images ... 4/15')
gen_5 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_0, 1)
print('... Generating images ... 5/15')
gen_6 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_0, 1)
print('... Generating images ... 6/15')
gen_7 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_0, 1)
print('... Generating images ... 7/15')
fak_12 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_12, 1)
print('... Generating images ... 8/15')
fak_25 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_12, 1)
print('... Generating images ... 9/15')
fak_37 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_12, 1)
print('... Generating images ... 10/15')
fak_50 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_12, 1)
print('... Generating images ... 11/15')
fak_62 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_12, 1)
print('... Generating images ... 12/15')
fak_75 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_12, 1)
print('... Generating images ... 13/15')
fak_87 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_12, 1)
print('... Generating images ... 14/15')
fak_100 = m.generate_samples_fn(cropped_images, np.random.randn(100, z_dim).astype('float32'), cap_12, 1)
print('... Generating images ... 15/15')

print('... Saving to disk')
for ix in range(bs):
    bg = Image.new("RGB", (512, 256), (255, 255, 255))
    bg.paste(to_image(cropped_images[ix]), (0, 0, 1*64, 64))
    bg.paste(to_image(gen_1[ix]), (1*64, 0, 2*64, 64))
    bg.paste(to_image(gen_2[ix]), (2*64, 0, 3*64, 64))
    bg.paste(to_image(gen_3[ix]), (3*64, 0, 4*64, 64))
    bg.paste(to_image(gen_4[ix]), (4*64, 0, 5*64, 64))
    bg.paste(to_image(gen_5[ix]), (5*64, 0, 6*64, 64))
    bg.paste(to_image(gen_6[ix]), (6*64, 0, 7*64, 64))
    bg.paste(to_image(gen_7[ix]), (7*64, 0, 8*64, 64))
    bg.paste(to_image(fak_12[ix]), (0*64, 2*64, 1*64, 3*64))
    bg.paste(to_image(fak_25[ix]), (1*64, 2*64, 2*64, 3*64))
    bg.paste(to_image(fak_37[ix]), (2*64, 2*64, 3*64, 3*64))
    bg.paste(to_image(fak_50[ix]), (3*64, 2*64, 4*64, 3*64))
    bg.paste(to_image(fak_62[ix]), (4*64, 2*64, 5*64, 3*64))
    bg.paste(to_image(fak_75[ix]), (5*64, 2*64, 6*64, 3*64))
    bg.paste(to_image(fak_87[ix]), (6*64, 2*64, 7*64, 3*64))
    bg.paste(to_image(fak_100[ix]), (7*64, 2*64, 8*64, 3*64))
    draw = ImageDraw.Draw(bg)
    draw.text((5,64+16), real[ix], (0,0,0))
    draw.text((5,3*64+16), fake[ix], (0,0,0))
    bg.save(os.path.join(SAVED_DIR, 'experiment-%03d.png') % (ix + 1))
    print('... Saved image %d/%d to disk' % (ix + 1, bs))



