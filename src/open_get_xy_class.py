from augmentation_class import Augmentation
from skimage.transform import resize
from skimage import io
import numpy as np
import glob
import os

class OpenGet(object):
    def __init__(self, path):
        self.path = path

    def open_images(self, path):
        '''open images, resize, perform grayscale, get edges, ravel'''
        color_images = io.imread(path)
        color_size = resize(color_images, (32, 32))
        ## rescale image as resize scales down the pixels
        rescaled_image = 255 * color_size
        ## Convert to integer data type pixels
        final_image = rescaled_image.astype(np.uint8)
        return final_image

    def get_X_y_fv(self, X, y, all_fru_veg, folder, grayscale):
        '''opens images and return an array'''
        ## updating this .py file to augment images in 20% subsets of each fv
        for fru_veg in all_fru_veg:
            label = fru_veg
            ## shuffle each fv image per folder, but after augment
            if folder == 'Train':
                path = glob.glob('data/Train/{}/*'.format(fru_veg))
                path_20 = int(len(path) * 0.20)
                path_40 = int(len(path) * 0.40)
                path_60 = int(len(path) * 0.60)
                path_80 = int(len(path) * 0.80)
                for p in path[:path_20 + 1]:
                    final_image = OpenGet.open_images(self, p)
                    X.append(Augmentation.gray_aug(self, final_image))
                    y.append(label)
                for p in path[path_20 + 1:path_40 + 1]:
                    final_image = OpenGet.open_images(self, p)
                    X.append(Augmentation.shift(self, final_image, direction='up', shift=10, roll=False))
                    y.append(label)
                for p in path[path_40 + 1:path_60 + 1]:
                    final_image = OpenGet.open_images(self, p)
                    ## put augment here
                    y.append(label)
                for p in path[path_60 + 1:path_80 + 1]:
                    final_image = OpenGet.open_images(self, p)
                    ## put augment here
                    y.append(label)
                for p in path[path_80 + 1:]:
                    final_image = OpenGet.open_images(self, p)
                    ## put augment here
                    y.append(label)
            elif folder == 'Test':
                path = glob.glob('data/Test/{}/*'.format(fru_veg))
                for p in path:
                    final_image = OpenGet.open_images(self, p)
                    X.append(final_image)
                    y.append(label)
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y

if __name__ == '__main__':
    pass    