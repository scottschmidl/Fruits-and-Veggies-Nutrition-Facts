from skimage import io
from skimage import filters
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import gaussian
import os
import matplotlib.pyplot as plt


def open_images(path, fruit_name, grayscale=False, edges=False):
    color_images = io.imread(path)   
    color_size = resize(color_images, (32, 32))
    if grayscale:
        gray_image = rgb2gray(color_size) # for making image gray scale
        gray_ravel = gray_image.ravel()  
        if edges:
            gray_edges = look_at_edges(fruit_name, gray_image) # for getting the edges 
            gray_ravel = gray_edges.ravel()           
    color_ravel = color_size.ravel()        
    return color_ravel, gray_ravel

def look_at_edges(fruit_name, size): 
    sobel_img = filters.sobel(size)
    io.imshow(sobel_img)
    plt.title('AVG {} Grayscale Edges'.format(fruit_name))
    plt.savefig('images/Avg_{}_color_edges.png'.format(fruit_name))
    plt.show()
    sobel_img.round(2)
    io.imshow(gaussian(sobel_img, sigma=3))
    plt.title('AVG {} Gaussian'.format(fruit_name))
    plt.savefig('images/avg_{}_gaussian.png'.format(fruit_name))
    plt.show()
    sobel_ravel = sobel_img.ravel()
    return sobel_ravel

if __name__ == '__main__':     
    tomato_img = open_images('data/fruits_vegetables/Tomato', 'Tomato')
    pear_img = open_images('data/fruits_vegetables/Pear', 'Pear')

    ## Featurization-Looking for edges in images
    plot_edges = look_at_edges('Tomato', size=None)
    plot_edges = look_at_edges('Pear', size=None)
    