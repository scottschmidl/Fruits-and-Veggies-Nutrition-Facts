from skimage.color import rgb2gray
from skimage import filters
import numpy as np 
 
class Augmentation(object):
    def __init__(self, final_image):
        self.final_image = final_image

    def gray_aug(self, final_image):
        gray_image = rgb2gray(final_image).ravel()                
        return gray_image
    
    def sobel_filter_aug(self, final_image):
        sobel_img = filters.sobel(final_image).ravel()
        return sobel_img

    def shift(self, final_image, direction, shift=10, roll=False):
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



if __name__ == '__main__':
    pass