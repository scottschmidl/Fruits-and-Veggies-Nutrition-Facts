from sklearn.metrics import (classification_report, plot_confusion_matrix,
                            plot_roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from image_loader import open_images
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def get_X_y_fv(X, y, all_fru_veg):
    for fru_veg in all_fru_veg:
        path = glob.glob('data/fruits_vegetables/{}/*.jpg'.format(fru_veg))
        label = fru_veg
        for p in path:
            X.append(open_images(p, '{}'.format(fru_veg)))
            y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, all_fru_veg

def roc_you_curve(X_train, X_test, y_train, y_test):
    '''get Receiver Operating Characteristic Curve'''
    model = MultinomialNB()
    model.fit(X_train, y_train)   
    plot_roc_curve(model, X_test, y_test, name='ROC Curve for Edge Images NB')
    plt.legend()
    plt.savefig('images/edge_roccurve.png',  bbox_inches='tight')
    plt.show()        
    return 

def plot_conf_matrix(X_train, X_test, y_train, y_test):
    '''get Confusion Matrix from NB'''
    model = MultinomialNB()
    model.fit(X_train, y_train)
    plot_confusion_matrix(model, X_test, y_test, labels=all_fru_veg, xticks_rotation=50)
    plt.title('Edge Confusion Matrix')
    plt.savefig('images/edge_confusion_matrix.png',  bbox_inches='tight')
    plt.show()
    return
    
def naive_bayes(X_train, X_test, y_train, y_test):
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
    X, y, all_fru_veg = get_X_y_fv(X, y, all_fru_veg)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    roc = roc_you_curve(X_train, X_test, y_train, y_test)      
    plot_conf_matrix = plot_conf_matrix(X_train, X_test, y_train, y_test)
    naiveb_model = naive_bayes(X_train, X_test, y_train, y_test)