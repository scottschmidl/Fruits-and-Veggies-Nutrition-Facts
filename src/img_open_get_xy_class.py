from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import filters
from skimage import io
import numpy as np
import glob
import os

class OpenGet(object):
    def __init__(self, grayscale, edge):
        self,grayscale = grayscale
        self.edge = edge

    def open_images(self, path):
        '''open images, resize, perform grayscale, get edges, ravel'''
        color_images = io.imread(path)
        color_size = resize(color_images, (32, 32))
        ## rescale image as resize scales down the pixels
        rescaled_image = 255 * color_size
        ## Convert to integer data type pixels
        final_image = rescaled_image.astype(np.uint8)
        return final_image
    
    def image_augmentation(self, final_image, grayscale, edge):
        ## need to add more augments to this
        if grayscale:
            ## for making image gray scale
            gray_image = rgb2gray(final_image)
            if edge:
                ## for getting the edges 
                sobel_img = filters.sobel(gray_image) 
                aug_image = sobel_img.ravel()
            else:
                aug_image = gray_image.ravel()           
        else:
            aug_image = final_image.ravel()      
        return aug_image
    
    ## Consider writing functions for every different augment

    def get_X_y_fv(self, X, y, all_fru_veg, folder, grayscale, edge):
        '''opens images and return an array'''
        ## updating this .py file to augment images in 25% subsets of each fv
        for fru_veg in all_fru_veg:
            ## shuffle each fv image per folder, but after augment
            if folder == 'fruits_vegetables':
                path = glob.glob('data/fruits_vegetables/{}/*'.format(fru_veg))                
            elif folder == 'Train':
                path = glob.glob('data/Train/{}/*'.format(fru_veg))
            elif folder == 'Test':
                path = glob.glob('data/Test/{}/*'.format(fru_veg))            
            label = fru_veg
            len_of_path = len(path)            
            for p in path:
                final_image = OpenGet.open_images(self, p)
                X.append(OpenGet.image_augmentation(self, final_image, grayscale, edge))                
                y.append(label)
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y, all_fru_veg

if __name__ == '__main__':
    pass    