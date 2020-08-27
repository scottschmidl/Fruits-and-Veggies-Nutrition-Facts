from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import resize
from EDA_img import look_at_edges
import os

def open_images(path, fruit_name):
    color_images = imread(path)
    gray_image = rgb2gray(color_images) #for making image gray scale
    # color_size = resize(color_images, (32, 32))
    # color_ravel = color_size.ravel()    
    gray_size = resize(gray_image, (32, 32))
    gray_edges = look_at_edges(fruit_name, gray_size)
    gray_ravel = gray_edges.ravel()    
    return gray_ravel

if __name__ == '__main__':     
    ## functions currently have grayscale active. see gray_image variable
    tomato_img = open_images('data/fruits_vegetables/Tomato', 'Tomato')
    # print(tomato_img, '\n')
    pear_img = open_images('data/fruits_vegetables/Pear', 'Pear')
    # print(pear_img, '\n')