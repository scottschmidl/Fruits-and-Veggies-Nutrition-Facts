from skimage import io
from skimage import filters
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import gaussian
import os
import matplotlib.pyplot as plt

def open_images(path, fruit_name, grayscale=False, edges=False):
    '''open images, resize, perform grayscale, get edges, ravel'''
    color_images = io.imread(path)   
    color_size = resize(color_images, (32, 32))
    if grayscale:
        gray_image = rgb2gray(color_size) # for making image gray scale
        if edges:
            gray_edges = look_at_edges(fruit_name, gray_image) # for getting the edges 
            ravel = gray_edges.ravel()
        else:
            ravel = gray_image.ravel()           
    else:
        ravel = color_size.ravel()        
    return ravel

def look_at_edges(fruit_name, grayimage): 
    '''function to get the edges of grayscaled images'''
    sobel_img = filters.sobel(grayimage)
    # io.imshow(sobel_img)
    # plt.title('AVG {} Grayscale Edges'.format(fruit_name))
    # plt.savefig('images/Avg_{}_color_edges.png'.format(fruit_name))
    # plt.show()
    # sobel_img.round(2)
    # io.imshow(gaussian(sobel_img, sigma=3))
    # plt.title('AVG {} Gaussian'.format(fruit_name))
    # plt.savefig('images/avg_{}_gaussian.png'.format(fruit_name))
    # plt.show()
    sobel_ravel = sobel_img.ravel()
    return sobel_ravel

if __name__ == '__main__':
    ## open up images   
    tomato_img = open_images('data/fruits_vegetables/Tomato', 'Tomato')
    pear_img = open_images('data/fruits_vegetables/Pear', 'Pear')

    ## Featurization-Looking for edges in images
    plot_edges = look_at_edges('Tomato', grayimage=None)
    plot_edges = look_at_edges('Pear', grayimage=None)
    