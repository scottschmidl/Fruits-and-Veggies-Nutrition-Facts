from augmentation import Augmentation
from skimage.transform import resize
from sklearn.utils import shuffle
from skimage import io
import numpy as np
import glob

class OpenGet:

    def __init__(self) -> None:
        self.X = []
        self.y = []

    def open_images(self, path):
        '''OPEN IMAGES, RESIZE, RAVEL'''
        color_images = io.imread(path)
        color_size = resize(color_images, (32, 32))
        ## rescale image as resize scales down the pixels
        rescaled_image = 255 * color_size
        ## Convert to integer data type pixels
        final_image = rescaled_image.astype(np.uint8)
        return final_image

    def get_X_y_fv(self, all_fru_veg, folder):
        '''OPENS IMAGES AND RETURN AN ARRAY'''
        ## updating this .py file to augment images in 20% subsets of each fv
        for fru_veg in all_fru_veg:
            label = fru_veg
            ## TODO: shuffle each fv image per folder, but after augment. Ensure X, y are shuffled the same way
            if folder == 'Train':
                path = glob.glob(f'data/Train/{fru_veg}/*')
                path_20 = int(len(path) * 0.20)
                path_40 = int(len(path) * 0.40)
                path_60 = int(len(path) * 0.60)
                path_80 = int(len(path) * 0.80)
                for p in path[:path_20 + 1]:
                    final_image = OpenGet.open_images(self, p)
                    self.X.append(Augmentation.rand_noise(self, final_image))
                    self.y.append(label)
                for p in path[path_20 + 1:path_40 + 1]:
                    final_image = OpenGet.open_images(self, p)
                    self.X.append(Augmentation.shift(self, final_image, direction='up', shift=10, roll=False))
                    self.y.append(label)
                for p in path[path_40 + 1:path_60 + 1]:
                    final_image = OpenGet.open_images(self, p)
                    self.X.append(Augmentation.hue_saturation(self, final_image))
                    self.y.append(label)
                for p in path[path_60 + 1:path_80 + 1]:
                    final_image = OpenGet.open_images(self, p)
                    self.X.append(final_image)
                    self.y.append(label)
                for p in path[path_80 + 1:]:
                    final_image = OpenGet.open_images(self, p)
                    self.X.append(Augmentation.sharpen(self, final_image))
                    self.y.append(label)
            elif folder == 'Test':
                path = glob.glob('data/Test/{}/*'.format(fru_veg))
                for p in path:
                    final_image = OpenGet.open_images(self, p)
                    self.X.append(final_image)
                    self.y.append(label)
        X = np.asarray(self.X)
        y = np.asarray(self.y)
        X, y = shuffle(X, y)
        return X, y
