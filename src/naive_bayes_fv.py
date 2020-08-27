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

def roc_you_curve(X, y, all_fru_veg): 
    model = MultinomialNB()
    model.fit(X_train, y_train)   
    plot_roc_curve(model, X_test, y_test, name='ROC Curve for Edge Images NB')
    plt.legend()
    plt.savefig('images/edge_roccurve.png',  bbox_inches='tight')
    plt.show()        
    return 'cookie monster'

def plot_conf_matrix(X, y, all_fru_veg):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    plot_confusion_matrix(model, X_test, y_test, labels=all_fru_veg, xticks_rotation=50)
    plt.title('Edge Confusion Matrix')
    plt.savefig('images/edge_confusion_matrix.png',  bbox_inches='tight')
    plt.show()
    return 'cookie monster'
    
def naive_bayes(X, y, all_fru_veg):
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

    # roc = roc_you_curve(X_train, X_test, y_train, y_test)
        
    # plot_conf_matrix = plot_conf_matrix(X_train, X_test, y_train, y_test)

    # naiveb_model = naive_bayes(X_train, X_test, y_train, y_test)

    # dimensionality reduction
    y_enumerated = []
    for fruit in y:
        if fruit == 'Tomato':
            y_enumerated.append(1)
        elif fruit == 'Pear':
            y_enumerated.append(0)
    y_enumerated = np.asarray(y_enumerated)

    pca = PCA(n_components=21)
    # pca.fit(X)
    # X_pca = pca.transform(X)
    # print("original shape:   ", X.shape)
    # print("transformed shape:", X_pca.shape)
    # # Light is original data points, dark is projected data points
    # # shows how much "information" is discarded in this reduction of dimensionality.
    # X_new = pca.inverse_transform(X_pca)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.1)
    # plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    # plt.axis('equal')
    # plt.show()

    projected = pca.fit_transform(X)
    print(X.data.shape)
    print(projected.shape)
    plt.scatter(projected[:, 0], projected[:, 1],
            c=y_enumerated, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('tab20c', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()  

    # print("There are {0} rows of data.".format(X.shape[0]), '\n')
    # look at a digit and target
    # img = 11
    # print(X[img])
    # print("\nThe images are {0} in shape.".format(X[img].shape))
    # print("\nEach value in the image is of type {0}".format(type(X[img][0])))
    # print("(Though they look a lot like 4 bit numbers.)")
    # # plt.imshow(X[img].reshape((32, 32)), cmap='gray')
    # # plt.show()
    # print("\nNumber in image: ", y[img])
    # not going to scale the images, because all pixel intensities
    # are already on the same scale

    # plot explained variance ratio in a scree plot
    # plt.figure(1, figsize=(8, 6))
    # plt.clf()
    # plt.axes([.2, .2, .7, .7])
    # plt.plot(pca.explained_variance_, linewidth=2, color='red')
    # plt.axis('tight')
    # plt.xlim(0, 50)
    # plt.xlabel('n_components')
    # plt.ylabel('explained_variance_')
    # plt.show()

    # total_variance = np.sum(pca.explained_variance_)
    # cum_variance = np.cumsum(pca.explained_variance_)
    # prop_var_expl = cum_variance/total_variance
    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.plot(prop_var_expl, color='red', linewidth=2, label='Explained variance')
    # ax.axhline(0.9, label='90% goal', linestyle='--', color="black", linewidth=1)
    # ax.set_xlim(0, 50)
    # ax.set_ylabel('cumulative prop. of explained variance')
    # ax.set_xlabel('number of principal components')
    # ax.legend()
    # plt.show()