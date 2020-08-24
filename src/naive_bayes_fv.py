from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, KFold
import glob
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                            multilabel_confusion_matrix, balanced_accuracy_score, classification_report)
import numpy as np
from image_loader import open_images
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

def get_X_y_fv():
    X = []
    y = []   
    y_enumerated = []                   
    data_folders = ['Train', 'Test']
    all_fru_veg = os.listdir('data/Train')
        
    for folder in data_folders:

        for idx, fruit_veg in enumerate(all_fru_veg):
            path = glob.glob('data/{}/{}/*.jpg'.format(folder, fruit_veg))
            label = fruit_veg
            
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

# def crossVal(k, threshold=0.50): # does not work with pca due to negatives
#     X, y = get_X_y_fv()[:2] #call from principal_component_analysis()?

#     kf = KFold(n_splits=k)
#     train_accuracy = []
#     test_accuracy = []
   
#     for train, test in kf.split(X):
#         # Split into train and test
#         X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
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

# def roc_you_curve(n_classes): # does not work with pca due to negatives, multiclass L-75
#     X, y = get_X_y_fv()[:2]  

#     # model = MultinomialNB()
    
#     X_train, X_test, y_train, y_test = train_test_split(X,y)
#     # model.fit(X_train, y_train)
#     # y_pred = model.predict(X_test) 
    
#     fpr = {}
#     tpr = {}
#     thresholds = {}

#     for i in range(n_classes): # try this for multiclass - IndexError: too many indices for array
#         fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_pred[:, i])        

#     # fpr, tpr, thresholds = roc_curve(y_test, y_pred) # not working due to multiclass.

#     # x = np.linspace(0,1, 100)
#     # _, ax = plt.subplots(1, figsize=(10,6))
#     # ax.plot(fpr, tpr, color='firebrick')
#     # ax.plot(x, x, linestyle='--', color ='black', label='Random Guess')
#     # ax.set_xlabel('False Positive Rate (FPR)', fontsize=16)
#     # ax.set_ylabel('True Positive Rate (TPR)', fontsize=16)
#     # ax.set_title('ROC Curve for Random Forest')
#     # plt.legend()
#     # plt.show()
#     # plt.savefig('../images/roccurve.png',  bbox_inches='tight')

#     return fpr, tpr, thresholds

# def non_negative_matrix_factorization():
#     X = get_X_y_fv()[0]
#     model = NMF()
#     W = model.fit_transform(X)
#     H = model.components_

#     return W, H

def naive_bayes():
    X, y, _, all_fru_veg = get_X_y_fv()

    X_train, X_test, y_train, y_test = train_test_split(X, y)    
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # acc = accuracy_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred, labels=all_fru_veg, average=None)
    # prec = precision_score(y_test, y_pred, labels=all_fru_veg, average=None)
    # fone = f1_score(y_test, y_pred, labels=all_fru_veg, average=None)
    mult_con_matrix = multilabel_confusion_matrix(y_test, y_pred, labels=all_fru_veg)
    # bal_acc_scor = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    return mult_con_matrix, report
    
if __name__ == '__main__':
    get_it = get_X_y_fv()
    # print(get_it)

    # cross = crossVal(5)

    # roc = roc_you_curve(len(get_it[2][:10])) # not working due to multiclass, can't use pca
                                               # due to negative values, 
                                               # IndexError: too many indices for array

    # nnmf = non_negative_matris_factorization()
    # print(nnmf)
 
    # naiveb_model = naive_bayes() # does not run due to negative values from pca, use get_it
    # print(naiveb_model)