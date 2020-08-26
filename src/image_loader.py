from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import resize
import os

def open_images(path):
    train_images = imread(path)
    gray_image = rgb2gray(train_images) #for making image gray scale
    color_size = resize(train_images, (32, 32))
    color_ravel = color_size.ravel()    
    gray_size = resize(gray_image, (32, 32))
    gray_ravel = gray_size.ravel()
    
    return color_ravel, gray_ravel

if __name__ == '__main__':     
    ## functions currently have grayscale active. see gray_image variable
    training_img = open_images('data/fruits_vegetables/Tomato')
    # print(training_img, '\n')
    testing_img = open_images('data/fruits_vegetables/Pear')
    # print(testing_img, '\n')
        