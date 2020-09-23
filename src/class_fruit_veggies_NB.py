from sklearn.metrics import (classification_report, plot_confusion_matrix,
                            plot_roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from image_loader import open_images
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import argparse
import pickle as pickle

class FruitsVeggiesNB(object):
    def __init__(self, X, y, all_fru_veg):
        self.X = X
        self.y = y
        self.all_fru_veg = all_fru_veg

    def get_X_y_fv(self, X, y, all_fru_veg, grayscale, edge):
        '''opens images and return an array'''
        for fru_veg in all_fru_veg:
            path = glob.glob('data/fruits_vegetables/{}/*'.format(fru_veg))
            label = fru_veg
            for p in path:
                X.append(open_images(p, '{}'.format(fru_veg), grayscale, edge))
                y.append(label)
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y, all_fru_veg
        
    def roc_you_curve(self, X_train, X_test, y_train, y_test, grayscale, edge): 
        '''returns Receiver Operating Characteristic Curve for NB'''
        model = MultinomialNB()
        model.fit(X_train, y_train)
        if edge and grayscale:
            name = 'ROC Curve for Edge Images NB'
        elif grayscale:
            name = 'ROC Curve for Grayscale Images NB'
        else:
            name = 'ROC Curve for Color Images NB'
        plot_roc_curve(model, X_test, y_test, name=name)
        plt.legend()
        if edge and grayscale:
                plt.savefig('images/edge_roccurve.png',  bbox_inches='tight')
        elif grayscale:
            plt.savefig('images/grayscale_roccurve.png',  bbox_inches='tight')            
        else:
            plt.savefig('images/color_roccurve.png',  bbox_inches='tight')
        plt.show()
        return plt

    def plot_conf_matrix(self, X_train, X_test, y_train, y_test, grayscale, edge):
        '''returns Confusion Matrix from NB'''
        model = MultinomialNB()
        model.fit(X_train, y_train)       
        plot_confusion_matrix(model, X_test, y_test, labels=all_fru_veg, xticks_rotation=50)
        if edge and grayscale:
                plt.title('Edge Confusion Matrix')
                plt.savefig('images/edge_confusion_matrix.png',  bbox_inches='tight')
        elif grayscale:
            plt.title('Grayscale Confusion Matrix')
            plt.savefig('images/grayscale_confusion_matrix.png',  bbox_inches='tight')            
        else:
            plt.title('Color Confusion Matrix')
            plt.savefig('images/color_confusion_matrix.png',  bbox_inches='tight')
        plt.show()
        return plt
    
    def random_forest(self, X_train, X_test, y_train, y_test, grayscale, edge):
        '''return Classification Report from RF'''
        model = RandomForestClassifier(n_estimators=25,
                                        criterion='gini',
                                        max_depth=10,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0,
                                        max_features='auto',
                                        bootstrap=True,
                                        n_jobs=None,
                                        random_state=None,
                                        verbose=1,
                                        max_samples=None)
        mod = model.fit(X_train, y_train)
        y_pred = mod.predict(X_test)
        report = classification_report(y_test, y_pred, digits=2)
        return mod, report
        
    def naive_bayes(self, X_train, X_test, y_train, y_test, grayscale, edge):
        '''returns Classification Report from NB'''
        model = MultinomialNB()
        mod = model.fit(X_train, y_train)
        y_pred = mod.predict(X_test)
        report = classification_report(y_test, y_pred, digits=2)
        return mod, report
    
if __name__ == '__main__':
    X = []
    y = []
    grayscale = True
    edge = True
    all_fru_veg = os.listdir('data/fruits_vegetables')
    fru_veg_class = FruitsVeggiesNB(X, y, all_fru_veg)
    X, y, all_fru_veg = fru_veg_class.get_X_y_fv(X, y, all_fru_veg, grayscale=grayscale, edge=edge)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    fru_veg_class.roc_you_curve(X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)        
    fru_veg_class.plot_conf_matrix(X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)
    rf_mod, report = fru_veg_class.random_forest(X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)
    print(report)
    # filename_rf = 'fv_app/fv_rf_model.sav'
    # pickle.dump(rf_mod, open(filename_rf, 'wb'))
    nb_mod, report = fru_veg_class.naive_bayes(X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)
    print(report)
    # filename_nb = 'fv_app/fv_nb_model.sav'
    # pickle.dump(nb_mod, open(filename_nb, 'wb'))    
