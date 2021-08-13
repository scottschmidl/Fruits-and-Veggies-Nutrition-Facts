from sklearn.metrics import (classification_report, plot_confusion_matrix,
                            plot_roc_curve)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imgsXy import OpenGet
import pickle as pkl
import xgboost as xgb
import os

class ModelsFruitsVeggies:

    def grid_search(self, X_train, y_train):
        '''
        RETURN BEST PARAMETERS FOR RANDOM FOREST
        '''
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
    def fit_the_models(self, model, X_train, y_train, pca_shape=None):
        '''
        FIT THE MODELS
        '''
        # THIS PART FITS FOR RF AND MNB
        if model == RandomForestClassifier():
            model_to_fit = RandomForestClassifier()
        elif model == MultinomialNB():
            model_to_fit = MultinomialNB()
        fit_model = model.fit(X_train, y_train)
        # THIS PART FITS FOR PCA
        if model == PCA():
            model_to_fit = PCA()
            fit_model = model_to_fit.fit(X=pca_shape)

        return fit_model

    def roc_you_curve(self, fit_model, X_test, y_test, grayscale, edge):
        '''
        RETURN RECEIVER OPERATing CHARACTERSTIC CURVE
        '''
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

    def plot_conf_matrix(self, fit_model, X_test, y_test, labels, edge, grayscale):
        '''
        RETURNS CONFUSION MATRIX
        '''
        plot_confusion_matrix(fit_model, X_test, y_test, labels=labels, xticks_rotation=50)
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

    def random_forest(self, fit_model, X_test, y_test):
        '''
        RETURNS CLASSIFICAITON REPORT FOR RF
        '''
        y_pred = fit_model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=2)
        return report

    def naive_bayes(self, fit_model, X_test, y_test):
        '''
        RETURNS CLASSIFICATION REPORT FOR NB
        '''
        y_pred = fit_model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=2)
        return report

def run_model_helper(model, X_train, X_test, y_train, y_test, all_train_fv, grayscale, edge):
    '''
    HELPER TO RUN THE MODELS
    '''
    ## instantiate class
    fru_veg_class = ModelsFruitsVeggies()
    print('fruits veggies class instantiated')
    ## models
    best_rf_model = fru_veg_class.grid_search(X_train, y_train)
    print("Random Forest best parameters:", best_rf_model)
    # TODO: run grid_search to find best parameters, before running below
    for m in model:
        fit_model = fru_veg_class.fit_the_models(m, X_train, y_train)
        fru_veg_class.roc_you_curve(fit_model, edge, grayscale, X_test, y_test)
        fru_veg_class.plot_conf_matrix(fit_model, X_test, y_test, all_train_fv, edge, grayscale)
        if m == RandomForestClassifier():
            rf_report = fru_veg_class.random_forest(fit_model, X_test, y_test)
            # print(report)
            # filename_rf = 'fv_app/fv_rf_model.sav'
            # pickle.dump(fit_model, open(filename_rf, 'wb'))
        elif m == MultinomialNB():
            nb_report = fru_veg_class.naive_bayes(fit_model, X_test, y_test)
            # print(report)
            # filename_nb = 'fv_app/fv_nb_model.sav'
            # pickle.dump(fit_model, open(filename_nb, 'wb'))

    return rf_report, nb_report

def main():
    '''
    RUN ALL CLASS INSTANCES AND MODELS
    '''
    ## paths to images
    all_train_fv = os.listdir('data/Train')
    all_test_fv = os.listdir('data/Test')
    ## use augments
    grayscale = False
    edge = False
    ## instaniate class
    open_get_class = OpenGet()
    ## open up images and get X_train, X_test, y_train, y_test
    X_train, y_train = open_get_class.get_X_y_fv(all_fru_veg=all_train_fv, folder='Train')
    print('this is x_train, y_train')
    X_test, y_test = open_get_class.get_X_y_fv(all_fru_veg=all_test_fv, folder='Test')
    print('this is x_test, y_test')
    #MODEL RUN
    models = [RandomForestClassifier(), MultinomialNB()]
    print('model')
    rf_report, nb_report = run_model_helper(models, X_train, X_test, y_train, y_test,
                                                        all_train_fv, grayscale, edge)
    print(f'Random Forest Classification Report:\n{rf_report}')
    print(f'Naive Bayes Classification Report:\n{nb_report}')

if __name__ == '__main__':
    main()