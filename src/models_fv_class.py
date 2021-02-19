from sklearn.metrics import (classification_report, plot_confusion_matrix,
                            plot_roc_curve)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from open_get_xy_class import OpenGet
import matplotlib.pyplot as plt
import pickle as pickle
import numpy as np
import argparse
import glob
import os

class ModelsFruitsVeggies():

    def __init__(self, X_train, X_test, y_train, y_test, grayscale, edge):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.grayscale = grayscale
        self.edge = edge

    def grid_search(self):
        '''return best parameters for random_forest'''
        random_forest_grid = {'n_estimators': [20, 40, 50, 100],
                            'criterion':['gini', 'entropy'],
                            'max_depth': list(range(1, 6)),
                            'min_samples_split': [2],
                            'min_samples_leaf': [1, 5],
                            'max_features': ['sqrt', 'log2', 'auto'],
                            'bootstrap': [True, False],
                            'random_state': [1]}
        rf_gridsearch = RandomizedSearchCV(RandomForestClassifier(),
                                    random_forest_grid,
                                    n_iter = 10,
                                    n_jobs=3,
                                    verbose=True,
                                    scoring='f1')
        rf_gridsearch.fit(X_train, y_train)
        best_rf_model = rf_gridsearch.best_estimator_
        return best_rf_model

    ## parameters of Random_Forest_Classifier will change after grid_search
    def fit_the_models(self, fit_model):
        if model == RandomForestClassifier():
            model = RandomForestClassifier()
        elif model == MultinomialNB():
            model = MultinomialNB()
        fit_model = model.fit(X_train, y_train)
        return fit_model

    def roc_you_curve(self, fit_model):
        '''returns Receiver Operating Characteristic Curve for NB'''
        if edge and grayscale:
            name = 'ROC Curve for Edge Images'
            plt.savefig('images/edge_roccurve.png',  bbox_inches='tight')
        elif grayscale:
            name = 'ROC Curve for Grayscale Images'
            plt.savefig('images/grayscale_roccurve.png',  bbox_inches='tight')
        else:
            name = 'ROC Curve for Color Images'
            plt.savefig('images/color_roccurve.png',  bbox_inches='tight')
        plot_roc_curve(fit_model, X_test, y_test, name=name)
        plt.legend()
        plt.show()
        return plt

    def plot_conf_matrix(self, fit_model):
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

    def random_forest(self, fit_model):
        '''return Classification Report from RF'''
        y_pred = fit_model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=2)
        return report

    def naive_bayes(self, fit_model):
        '''returns Classification Report from NB'''
        y_pred = fit_model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=2)
        return report

def main():
    ## instantiate class
    fru_veg_class = ModelsFruitsVeggies(X_train, X_test, y_train, y_test, grayscale, edge)
    print('fruits veggies class instantiated')
    ## models
    best_rf_model = fru_veg_class.grid_search()
    print("Random Forest best parameters:", best_rf_model)
    model = [RandomForestClassifier(), MultinomialNB()]
    print('model')
    # TODO: run grid_search to find best parameters, before running below
    # for m in model:
    #     fit_model = fru_veg_class.fit_the_models(fit_model=m)
    #     fru_veg_class.roc_you_curve(fit_model)
    #     fru_veg_class.plot_conf_matrix(fit_model)
    #     if m == RandomForestClassifier():
    #         rf_mod, report = fru_veg_class.random_forest(fit_model)
    #         # print(report)
    #         # filename_rf = 'fv_app/fv_rf_model.sav'
    #         # pickle.dump(fit_model, open(filename_rf, 'wb'))
    #     elif m == MultinomialNB():
    #         nb_mod, report = fru_veg_class.naive_bayes(fit_model)
    #         # print(report)
    #         # filename_nb = 'fv_app/fv_nb_model.sav'
    #         # pickle.dump(fit_model, open(filename_nb, 'wb'))

if __name__ == '__main__':
    ## paths to images
    all_train_fv = os.listdir('data/Train')
    all_test_fv = os.listdir('data/Test')
    ## instantiate lists
    X = []
    y = []
    ## use augments
    grayscale = False
    edge = False
    ## instaniate class
    open_get_class = OpenGet(X, y)
    ## open up images and get X_train, X_test, y_train, y_test
    X_train, y_train = open_get_class.get_X_y_fv(all_fru_veg=all_train_fv, folder='Train')
    print('this is x_train, y_train')
    X_test, y_test = open_get_class.get_X_y_fv(all_fru_veg=all_test_fv, folder='Test')
    print('this is x_test, y_test')
    main()