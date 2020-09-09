from sklearn.metrics import (classification_report, plot_confusion_matrix,
                            plot_roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import io

def open_images(path, fruit_name, grayscale=False, edges=False):
    '''open images, resize, perform grayscale, and ravel'''
    color_images = io.imread(path)   
    color_size = resize(color_images, (32, 32))
    if grayscale:
        gray_image = rgb2gray(color_size) # for making image gray scale
        ravel = gray_image.ravel()           
    else:
        ravel = color_size.ravel()        
    return ravel

class FruitsVeggiesNB(object):
    def __init__(self, X, y, all_fru_veg):
        self.X = X
        self.y = y
        self.all_fru_veg = all_fru_veg

    def get_X_y_fv(self, X, y, all_fru_veg, grayscale, edge):
        '''gets images and places into array'''
        for fru_veg in all_fru_veg:
            path = glob.glob('data/fruits_vegetables/{}/*'.format(fru_veg))
            label = fru_veg
            for p in path:
                X.append(open_images(p, '{}'.format(fru_veg), grayscale, edge))
                y.append(label)
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y, all_fru_veg
        
    def naive_bayes(self, X_train, X_test, y_train, y_test, grayscale, edge):
        '''get Classification Report from NB'''
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=3)
        return report
    
if __name__ == '__main__':
    pass

