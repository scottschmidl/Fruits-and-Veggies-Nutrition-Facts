
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, confusion_matrix,
                            multilabel_confusion_matrix, balanced_accuracy_score, classification_report,
                            plot_confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB 
from sklearn.decomposition import NMF
from image_loader import open_images
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

class fruits_veggies(object):
    def __init__(self, all_fru_veg):
        self.all_fru_veg = all_fru_veg


    def get_X_y_fv():
        X = []
        y = []   
        y_enumerated = []

        all_fru_veg = os.listdir('data/fruits_vegetables')[10:41:27]
        
        for idx, fru_veg in enumerate(all_fru_veg):
            
            path = glob.glob('data/fruits_vegetables/{}/*.jpg'.format(fru_veg))
            label = fru_veg
            
            for p in path:
                X.append(open_images(p))
                y.append(label)
                y_enumerated.append(idx)

        X = np.asarray(X)
        # print(len(X), '\n')
        # print('*********************************************\n')
        y = np.asarray(y)
        # print(len(y))
        
        return X, y, y_enumerated, all_fru_veg

    # def crossVal(k, threshold=0.50): 
    #     X, y_enumerated = get_X_y_fv()[:4:2] 

    #     kf = KFold(n_splits=k)
    #     train_accuracy = []
    #     test_accuracy = []
    
    #     for train, test in kf.split(X):
    #         # Split into train and test
    #         X_train, X_test, y_train, y_test = X[train], X[test], y_enumerated[train], y_enumerated[test]
    #         # Fit estimator
    #         model = MultinomialNB()
    #         model.fit(X_train, y_train)
    #         # Measure performance
    #         y_hat_trainprob = model.predict_proba(X_train)[:,1]
    #         y_hat_testprob = model.predict_proba(X_test)[:,1]
    #         y_hat_train = (y_hat_trainprob >= threshold).astype(int)
    #         y_hat_test = (y_hat_testprob >= threshold).astype(int)
    #         # metrics
    #         train_accuracy.append(accuracy_score(y_train, y_hat_train))
    #         test_accuracy.append(accuracy_score(y_test, y_hat_test))
    #     return np.mean(train_accuracy), np.mean(test_accuracy)

    def roc_you_curve(n_classes, X_test, X_train, y_test, y_train): 
        X, y_enumerated = get_X_y_fv()[:4:2]  

        

        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test) 
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        x = np.linspace(0,1, 100)
        _, ax = plt.subplots(1, figsize=(10,6))
        ax.plot(fpr, tpr, color='firebrick')
        ax.plot(x, x, linestyle='--', color ='black', label='Random Guess')
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=16)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=16)
        ax.set_title('ROC Curve for Random Forest')
        plt.legend()
        # plt.savefig('/images/roccurve.png',  bbox_inches='tight')
        plt.show()
        
        return fpr, tpr, thresholds

    # def non_negative_matrix_factorization():
    #     X = get_X_y_fv()[0]
    #     model = NMF()
    #     W = model.fit_transform(X)
    #     H = model.components_

    #     return W, H

    def plot_conf_matrix():
        X, y = get_X_y_fv()[0:2]
        all_fru_veg = get_X_y_fv()[3]

        

        model = MultinomialNB()
        model.fit(X_train, y_train)

        plot_confusion_matrix(model, X_test, y_test, labels=all_fru_veg, xticks_rotation=60)
        # plt.savefig('/images/confusion_matrix.png',  bbox_inches='tight')
        plt.show()
        
    def naive_bayes(X, y, all_fru_veg):
        X, y, _, all_fru_veg = get_X_y_fv()
        
            
        
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acc = accuracy_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred, labels=all_fru_veg, average=None)
        # prec = precision_score(y_test, y_pred, labels=all_fru_veg, average=None)
        # fone = f1_score(y_test, y_pred, labels=all_fru_veg, average=None)
        # conf_mat = confusion_matrix(y_test, y_pred, labels=all_fru_veg)
        # mult_con_matrix = multilabel_confusion_matrix(y_test, y_pred, labels=all_fru_veg)
        # bal_acc_scor = balanced_accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3)

        return report
    
if __name__ == '__main__':
    get_it = fruits_veggies.get_X_y_fv()
    X, y, _, all_fru_veg = get_it[]
    X, y_enumerated = get_it[]
    all_fru_veg = get_it[]

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_enumerated)
    X_train2, X_test2, y_test2, y_train2 = train_test_split(X, y)
    
    # print(get_it)

    # cross = crossVal(5)

    roc = fruits_veggies.roc_you_curve(2)

    # nnmf = non_negative_matris_factorization()
    # print(nnmf)

    plot = fruits_veggies.plot_conf_matrix()
 
    naiveb_model = fruits_veggies.naive_bayes() # does not run due to negative values from pca, use get_it
    # print(naiveb_model)