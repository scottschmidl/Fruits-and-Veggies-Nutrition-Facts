from skimage import io
from skimage import filters
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import gaussian
import os
import matplotlib.pyplot as plt
from eda_img import scale_fruit
import numpy as np

def open_images(path, fruit_name, grayscale, edges):
    '''open images, resize, perform grayscale, get edges, ravel'''
    color_images = io.imread(path)
    color_size = resize(color_images, (32, 32))
    ## rescale image as it scales down the pixels
    rescaled_image = 255 * color_size
    ## Convert to integer data type pixels.
    final_image = rescaled_image.astype(np.uint8)
    if grayscale:
        gray_image = rgb2gray(final_image) ## for making image gray scale
        if edges:
            sobel_img = filters.sobel(gray_image) ## for getting the edges 
            ravel = sobel_img.ravel()
        else:
            ravel = gray_image.ravel()           
    else:
        ravel = final_image.ravel()      
    return ravel

def look_at_edges(fruit_name, grayimage): 
    '''function to get the edges of grayscaled images'''
    sobel_img = filters.sobel(grayimage)
    io.imshow(sobel_img)
    plt.title('{} Grayscale Edges'.format(fruit_name))
    plt.savefig('images/{}_gray_edges.png'.format(fruit_name))
    plt.show()
    sobel_img.round(2)
    io.imshow(gaussian(sobel_img, sigma=3))
    plt.title('{} Gaussian'.format(fruit_name))
    plt.savefig('images/_{}_gaussian.png'.format(fruit_name))
    plt.show()
    sobel_ravel = sobel_img.ravel()
    return sobel_ravel

if __name__ == '__main__':
    grayscale = False
    edges = False
    ## open up images   
    tomato_img = open_images('data/fruits_vegetables/Tomato', 'Tomato', grayscale, edges)
    pear_img = open_images('data/fruits_vegetables/Pear', 'Pear', grayscale, edges)

    ## Featurization-Looking for edges in images
    plot_edges = look_at_edges('Tomato', grayimage=None)
    plot_edges = look_at_edges('Pear', grayimage=None)
    