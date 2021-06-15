import imgaug.augmenters as iaa
from skimage.util import random_noise
from skimage.color import rgb2gray
from skimage import filters
import numpy as np

class Augmentation:

    def gray_aug(self, final_image):
        gray_image = rgb2gray(final_image).ravel()
        return gray_image

    def sobel_filter_aug(self, final_image):
        sobel_img = filters.sobel(final_image).ravel()
        return sobel_img

    def hue_saturation(self, final_image):
        hue_sat_img = iaa.AddToHueAndSaturation(value=[-255, 255], value_hue=None, value_saturation=None,
                                                    per_channel=True, seed=50)
        hue_sat_img = hue_sat_img.augment_image(image=final_image)
        return hue_sat_img

    def sharpen(self, final_image):
        sharper = iaa.Sharpen(alpha=[0.25, 0.75], lightness=[1.0, 1.5], seed=50)
        sharper = sharper.augment_image(image=final_image)
        return sharper

    def rand_noise(self, final_image):
        ran_noise = random_noise(final_image, mode='gaussian', seed=50)
        return ran_noise

    def shift(self, final_image, direction, shift=10, roll=False):
        if direction == 'right':
            final_image[:, shift:] = final_image[:, :-shift]
            if roll:
                final_image[:,:shift] = np.fliplr(final_image[:, -shift:])
        elif direction == 'left':
            final_image[:, :-shift] = final_image[:, shift:]
            if roll:
                final_image[:, -shift:] = final_image[:, :shift]
        elif direction == 'down':
            final_image[shift:, :] = final_image[:-shift,:]
            if roll:
                final_image[:shift, :] = final_image[-shift:, :]
        elif direction == 'up':
            final_image[:-shift, :] = final_image[shift:, :]
            if roll:
                final_image[-shift:,:] = final_image[:shift, :]
        return final_image
