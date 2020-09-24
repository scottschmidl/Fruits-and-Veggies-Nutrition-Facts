from sklearn.metrics import (classification_report, plot_confusion_matrix,
                            plot_roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from img_open_get_xy_class import OpenGet
import matplotlib.pyplot as plt
import pickle as pickle
import numpy as np
import argparse
import glob
import os

class ModelsFruitsVeggies(object):
    def __init__(self, X_train, X_test, y_train, y_test, grayscale, edge):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.grayscale = grayscale
        self.edge = edge
        
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
    ## paths to images
    all_fru_veg = os.listdir('data/fruits_vegetables')
    all_train_fv = os.listdir('data/Train')
    all_test_fv = os.listdir('data/Test')
    ## use grayscale/edge
    grayscale = False
    edge = False
    ## instantiate class 
    open_get_class = OpenGet(grayscale=grayscale, edge=edge) 
    ## open images and ravel
    # open_get_train = open_get_class.open_images(path=all_train_fv, grayscale=grayscale, edge=edge)
    # open_get_test = open_get_class.open_images(path=all_test_fv, grayscale=grayscale, edge=edge)
    ## instantiate lists for getting arrays
    X = []
    y = []       
    ## open up images and get X_train, X_test, y_train, y_test
    all = True
    train = False
    test = False
    if all:
        X, y, all_fru_veg = open_get_class.get_X_y_fv(X=X, y=y, all_fru_veg=all_fru_veg, folder='fruits_vegetables', grayscale=grayscale, edge=edge) 
        X_train, X_test, y_train, y_test = train_test_split(X, y)
    elif train:
        X_train, y_train, _ = open_get_class.get_X_y_fv(X=X, y=y, all_fru_veg=all_train_fv, folder='Train', grayscale=grayscale, edge=edge)
    elif test:
        X_test, y_test, _ = open_get_class.get_X_y_fv(X=X, y=y, all_fru_veg=all_test_fv, folder='Test', grayscale=grayscale, edge=edge)    
    ## instantiate class
    fru_veg_class = ModelsFruitsVeggies(X_train, X_test, y_train, y_test, grayscale, edge)
    ## models
    fru_veg_class.roc_you_curve(X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)        
    fru_veg_class.plot_conf_matrix(X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)
    rf_mod, report = fru_veg_class.random_forest(X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)
    # print(report)
    # filename_rf = 'fv_app/fv_rf_model.sav'
    # pickle.dump(rf_mod, open(filename_rf, 'wb'))
    nb_mod, report = fru_veg_class.naive_bayes(X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)
    # print(report)
    # filename_nb = 'fv_app/fv_nb_model.sav'
    # pickle.dump(nb_mod, open(filename_nb, 'wb'))   
    