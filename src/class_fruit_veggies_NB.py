from sklearn.metrics import (classification_report, plot_confusion_matrix,
                            plot_roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from image_loader import open_images
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

class FruitsVeggiesNB(object):
    def __init__(self, X, y, all_fru_veg):
        self.X = X
        self.y = y
        self.all_fru_veg = all_fru_veg

    def get_X_y_fv(self, X, y, all_fru_veg):
        '''gets images and places into array'''
        for fru_veg in all_fru_veg:
            path = glob.glob('data/fruits_vegetables/{}/*.jpg'.format(fru_veg))
            label = fru_veg
            for p in path:
                X.append(open_images(p, '{}'.format(fru_veg)))
                y.append(label)
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y, all_fru_veg

    def roc_you_curve(self, X_train, X_test, y_train, y_test): 
        '''get Receiver Operating Characteristic Curve for NB'''
        model = MultinomialNB()
        model.fit(X_train, y_train)   
        plot_roc_curve(model, X_test, y_test, name='ROC Curve for Edge Images NB')
        plt.legend()
        plt.savefig('images/edge_roccurve.png',  bbox_inches='tight')
        plt.show()
        return plt

    def plot_conf_matrix(self, X_train, X_test, y_train, y_test):
        '''get Confusion Matrix from NB'''
        model = MultinomialNB()
        model.fit(X_train, y_train)
        plot_confusion_matrix(model, X_test, y_test, labels=all_fru_veg, xticks_rotation=50)
        plt.title('Edge Confusion Matrix')
        plt.savefig('images/edge_confusion_matrix.png',  bbox_inches='tight')
        plt.show()
        return plt
        
    def naive_bayes(self, X_train, X_test, y_train, y_test):
        '''get Classification Report from NB'''
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=3)
        return report
    
if __name__ == '__main__':
    X = []
    y = []   
    all_fru_veg = os.listdir('data/fruits_vegetables')[10:41:27]    
    fru_veg_class = FruitsVeggiesNB(X, y, all_fru_veg)
    X, y, all_fru_veg = fru_veg_class.get_X_y_fv(X, y, all_fru_veg)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    roc = fru_veg_class.roc_you_curve(X_train, X_test, y_train, y_test)        
    plot_conf_matrix = fru_veg_class.plot_conf_matrix(X_train, X_test, y_train, y_test)
    naiveb_model = fru_veg_class.naive_bayes(X_train, X_test, y_train, y_test)
    print(naiveb_model)