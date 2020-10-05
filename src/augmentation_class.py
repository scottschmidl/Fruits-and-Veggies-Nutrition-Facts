from skimage.color import rgb2gray
from skimage import filters
import numpy as np 
 
class Augmentation(object):
    def __init__(self, grayscale, edge):
        self.grayscale = grayscale
        self.edge = edge

    def gray_aug(self, final_image):
        gray_image = rgb2gray(final_image)
        aug_image = gray_image.ravel()                
        return aug_image
    
    def sobel_filter_aug(self, final_image):
        sobel_img = filters.sobel(final_image)
        aug_image = sobel_img.ravel()
        return aug_image

    def translate(self, final_image, direction, shift=10, roll=False):
        if direction == 'right':
            final_image[:, shift:] = final_image[:, :-shift]
            if roll:
                final_image[:,:shift] = np.fliplr(final_image[:, -shift:])
        if direction == 'left':
            final_image[:, :-shift] = final_image[:, shift:]
            if roll:
                final_image[:, -shift:] = final_image[:, :shift]
        if direction == 'down':
            final_image[shift:, :] = final_image[:-shift,:]
            if roll:
                final_image[:shift, :] = final_image[-shift:, :]
        if direction == 'up':
            final_image[:-shift, :] = final_image[shift:, :]
            if roll:
                final_image[-shift:,:] = final_image[:shift, :]
        return final_image