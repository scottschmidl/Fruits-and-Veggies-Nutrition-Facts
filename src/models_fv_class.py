from sklearn.metrics import (classification_report, plot_confusion_matrix,
                            plot_roc_curve)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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
        
    def grid_search(self, X_train, y_train):
        '''return best parameters for random_forest'''
        random_forest_grid = {'max_depth': [3, 5, None],
                            'max_features': ['sqrt', 'log2', None],
                            'min_samples_split': [2, 4],
                            'min_samples_leaf': [1, 5, 10, 15],
                            'bootstrap': [True, False],
                            'n_estimators': [20, 40, 50, 100, 200],
                            'random_state': [1]}
        rf_gridsearch = RandomizedSearchCV(RandomForestClassifier(),
                                    random_forest_grid,
                                    n_iter = 200,
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring='accuracy')
        rf_gridsearch.fit(X_train, y_train)
        best_rf_model = rf_gridsearch.best_estimator_
        return best_rf_model

    def fit_the_models(self, model, X_train, y_train):
        if model == RandomForestClassifier():
            model = RandomForestClassifier()
        elif model == MultinomialNB():
            model = MultinomialNB()
        fit_model = model.fit(X_train, y_train)
        return fit_model

    def roc_you_curve(self, fit_model, X_train, X_test, y_train, y_test, grayscale, edge): 
        '''returns Receiver Operating Characteristic Curve for NB'''
        if edge and grayscale:
            name = 'ROC Curve for Edge Images NB'
        elif grayscale:
            name = 'ROC Curve for Grayscale Images NB'
        else:
            name = 'ROC Curve for Color Images NB'
        plot_roc_curve(fit_model, X_test, y_test, name=name)
        plt.legend()
        if edge and grayscale:
                plt.savefig('images/edge_roccurve.png',  bbox_inches='tight')
        elif grayscale:
            plt.savefig('images/grayscale_roccurve.png',  bbox_inches='tight')            
        else:
            plt.savefig('images/color_roccurve.png',  bbox_inches='tight')
        plt.show()
        return plt

    def plot_conf_matrix(self, fit_model, X_train, X_test, y_train, y_test, grayscale, edge):
        '''returns Confusion Matrix from NB'''      
        plot_confusion_matrix(fit_model, X_test, y_test, labels=all_test_fv, xticks_rotation=50)
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
    
    ## parameters of Random_Forest_Classifier will change after grid_search
    def random_forest(self, mod, X_test, y_test, grayscale, edge):
        '''return Classification Report from RF'''
        y_pred = mod.predict(X_test)
        report = classification_report(y_test, y_pred, digits=2)
        return mod, report
        
    def naive_bayes(self, mod, X_test, y_test, grayscale, edge):
        '''returns Classification Report from NB'''
        y_pred = mod.predict(X_test)
        report = classification_report(y_test, y_pred, digits=2)
        return mod, report
    
if __name__ == '__main__':
    ## instantiate class 
    open_get_class = OpenGet(grayscale=grayscale, edge=edge)
    ## paths to images
    all_train_fv = os.listdir('data/Train')
    all_test_fv = os.listdir('data/Test')
    ## use augments
    grayscale = False
    edge = False 
    ## instantiate lists for getting arrays
    X = []
    y = []       
    ## open up images and get X_train, X_test, y_train, y_test
    X_train, y_train, _ = open_get_class.get_X_y_fv(X=X, y=y, all_fru_veg=all_train_fv, folder='Train', grayscale=grayscale, edge=edge)
    X_test, y_test, _ = open_get_class.get_X_y_fv(X=X, y=y, all_fru_veg=all_test_fv, folder='Test', grayscale=grayscale, edge=edge)    
    ## instantiate class
    fru_veg_class = ModelsFruitsVeggies(X_train, X_test, y_train, y_test, grayscale, edge)
    ## models
    model = [RandomForestClassifier(), MultinomialNB()]
    best_rf_model = fru_veg_class.grid_search(X_train, y_train)
    print("Random Forest best parameters:", best_rf_model)
    '''run grid_search before below'''
    for m in model:
        fit_model = fru_veg_class.fit_the_models(model=m, X_train=X_train, y_train=y_train)
        fru_veg_class.roc_you_curve(fit_model, X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)        
        fru_veg_class.plot_conf_matrix(fit_model, X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)
        if m == RandomForestClassifier():
            rf_mod, report = fru_veg_class.random_forest(fit_model, X_test, y_test, grayscale=grayscale, edge=edge)
            # print(report)
            # filename_rf = 'fv_app/fv_rf_model.sav'
            # pickle.dump(rf_mod, open(filename_rf, 'wb'))
        elif m == MultinomialNB():
            nb_mod, report = fru_veg_class.naive_bayes(fit_model, X_test, y_test, grayscale=grayscale, edge=edge)
            # print(report)
            # filename_nb = 'fv_app/fv_nb_model.sav'
            # pickle.dump(nb_mod, open(filename_nb, 'wb'))   
    