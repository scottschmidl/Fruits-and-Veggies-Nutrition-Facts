from open_get_xy_class import OpenGet
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

class PCAFruitsVeggies(object):
    def __init__(self, X, pca, total_variance, cum_variance, prop_var_expl):
        self.X = X
        self.pca = pca
        self.total_variance = total_variance
        self.cum_variance = cum_variance
        self.prop_var_expl = prop_var_expl        

    def scree_plot(self):
        '''get a Scree Plot to find number of components'''
        # plot explained variance ratio in a scree plot
        plt.figure(1, figsize=(8, 6))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(pca.explained_variance_, linewidth=2, color='red')
        plt.axis('tight')
        plt.xlim(0, 150)
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.savefig('images/scree_plot.png')
        plt.show()
        return plt

    def variance_explained(self): 
        '''better visualization of Scree Plot'''
        _, ax = plt.subplots(figsize=(8,6))
        ax.plot(prop_var_expl, color='red', linewidth=2, label='Explained Variance')
        ax.axhline(0.9, label='90% goal', linestyle='--', color="black", linewidth=1)
        ax.set_xlim(0, 150)
        ax.set_ylabel('Cumulative Prop. Of Explained Variance')
        ax.set_xlabel('Number Of Principal Components')
        ax.legend()
        plt.savefig('images/variance_explained.png')
        plt.show()
        return plt

    def pca_plot(self, list_of_colors):
        '''gets the dimensionality reducation and plots'''
        X_pca = pca.transform(X)
        # Light is original data points, dark is projected data points
        # shows how much "information" is discarded in this reduction of dimensionality.
        X_new = pca.inverse_transform(X_pca)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
        plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
        plt.axis('equal')
        plt.savefig('images/information_discard.png')
        plt.show()
        
        projected = pca.fit_transform(X)
        plt.scatter(projected[:, 0], projected[:, 1],
                c=list_of_colors, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('seismic', 5))
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar()
        plt.savefig('images/the_data_for_86_components_kept')
        plt.show()
        return plt

if __name__ == '__main__':
    X = []
    y = []
    folder = 'Train'
    grayscale = False
    edge = False
    all_train_fv = os.listdir('data/Train')
    open_get_class = OpenGet(X, y, grayscale)
    X, _ = open_get_class.get_X_y_fv(all_train_fv, folder)
    pca = PCA()
    pca.fit(X=(78756, 138))
    ## calculations for modles
    total_variance = np.sum(pca.explained_variance_)
    cum_variance = np.cumsum(pca.explained_variance_)
    prop_var_expl = cum_variance/total_variance
    list_of_colors = list(range(138))
    ## models
    fru_veg_pca = PCAFruitsVeggies(X, pca, total_variance, cum_variance, prop_var_expl)
    screech = fru_veg_pca.scree_plot()
    var_exp = fru_veg_pca.variance_explained()
    # plot_pca = fru_veg_pca.pca_plot(list_of_colors)
