from skimage.io import imread, imshow
from skimage import filters
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import gaussian
import os

def open_images(path, fruit_name):
    color_images = imread(path)
    gray_image = rgb2gray(color_images) # for making image gray scale
    # color_size = resize(color_images, (32, 32))
    # color_ravel = color_size.ravel()    
    gray_size = resize(gray_image, (32, 32))
    # gray_edges = look_at_edges(fruit_name, gray_size) # for getting the edges
    gray_ravel = gray_size.ravel()    
    return gray_ravel

def look_at_edges(fruit_name, size): 
    # fruit_name_gray = scale_fruit(fruit_name)[0]       
    sobel_img = filters.sobel(size)
    # io.imshow(sobel_img)
    # plt.title('AVG {} Grayscale Edges'.format(fruit_name))
    # plt.savefig('images/Avg_{}_color_edges.png'.format(fruit_name))
    # plt.show()
    # sobel_img.round(2)
    # io.imshow(gaussian(sobel_img, sigma=3))
    # plt.title('AVG {} Gaussian'.format(fruit_name))
    # plt.savefig('images/avg_{}_gaussian.png'.format(fruit_name))
    # plt.show()    
    return sobel_img

if __name__ == '__main__':     
    ## function currently has grayscale active.
    tomato_img = open_images('data/fruits_vegetables/Tomato', 'Tomato')
    pear_img = open_images('data/fruits_vegetables/Pear', 'Pear')

    ## Featurization-Looking for edges in images
    plot_edges = look_at_edges('Tomato', size=None)
    plot_edges = look_at_edges('Pear', size=None)
    