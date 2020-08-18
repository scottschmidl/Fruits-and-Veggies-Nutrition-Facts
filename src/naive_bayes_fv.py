from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import numpy as np
from image_loader import open_images
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pickle

# def rmse(y_true, y_predict):
#     return np.sqrt(mean_squared_error(y_true, y_predict))

def get_X_y():
    X = []
    y = []   
                   
    data_folders = ['Training', 'Testing']

    #function fails when all_fru_veg calls every folder..requires indexing
    all_fru_veg = os.listdir('data/Training')

    for folder in data_folders:
        
        for fruit_veg in all_fru_veg:
            path = glob.glob('data/{}/{}/*.jpg'.format(folder, fruit_veg))
            label = fruit_veg
            for p in path:                
                X.append(open_images(p))
                y.append(label)
    
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y, all_fru_veg

# def crossVal(k, threshold=0.50):
#     kf = KFold(n_splits=k)
#     train_accuracy = []
#     test_accuracy = []
#     X, y, _ = get_X_y()
   
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
#         #metrics
#         train_accuracy.append(accuracy_score(y_train, y_hat_train))
#         test_accuracy.append(accuracy_score(y_test, y_hat_test))
#     return np.mean(train_accuracy), np.mean(test_accuracy) #np.mean(test_errors), np.mean(train_errors), np.mean(test_r2), np.mean(train_r2)

# def roc_you_curve():
#     X, y, _ = get_X_y()

#     model = MultinomialNB()
    
#     X_train, X_test, y_train, y_test = train_test_split(X,y)
#     model.fit(X_train,y_train)
#     probabilities = model.predict_proba(X_test)[:,1]

#     fpr, tpr, thresholds = roc_curve(y_test, probabilities)
#     x = np.linspace(0,1, 100)
#     _, ax = plt.subplots(1, figsize=(10,6))
#     ax.plot(fpr, tpr, color='firebrick')
#     ax.plot(x, x, linestyle='--', color ='black', label='Random Guess')
#     ax.set_xlabel('False Positive Rate (FPR)', fontsize=16)
#     ax.set_ylabel('True Positive Rate (TPR)', fontsize=16)
#     ax.set_title('ROC Curve for Random Forest')
#     plt.legend()
#     #plt.savefig('../images/roccurve.png',  bbox_inches='tight')

#     return thresholds[fpr>0.3][0]

def naive_bayes():
    X, y, all_fru_veg = get_X_y()

    X_train, X_test, y_train, y_test = train_test_split(X, y)    
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # acc = accuracy_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred, labels=all_fru_veg, average=None)
    # prec = precision_score(y_test, y_pred, labels=all_fru_veg, average=None)
    fone = f1_score(y_test, y_pred, labels=all_fru_veg, average=None)
    con_matrix = confusion_matrix(y_test, y_pred)
    # bal_acc_scor = balanced_accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred, digits=3)

    return fone, con_matrix
    
if __name__ == '__main__':
    # cross = crossVal(5)

    # roc = roc_you_curve()
    
    naiveb_model = naive_bayes()
    print(naiveb_model)
    
