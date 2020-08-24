from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import resize
import os

def open_images(path):
    train_images = imread(path)
    gray_image = rgb2gray(train_images) #for making image gray scale
    # color_size = resize(train_images, (32, 32))    
    gray_size = resize(gray_image, (32, 32))
    gray_ravel = gray_size.ravel()
    
    return gray_ravel

# get_file_names and get_jpgs was part of my EDA   
def get_file_names(dir_name):
    # create a list of file and sub directories 
    # names in the given directory 
    list_of_file = os.listdir(dir_name)
    all_files = []
    # Iterate over all the entries
    for entry in list_of_file:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(full_path):
            all_files = all_files + get_file_names(full_path)
        else:
            all_files.append(full_path)
    return all_files

def get_jpgs(t_set):
    images_array = []
    for path in t_set:
        if path[-3:] == 'jpg':
            images_array.append(open_images(path))
    return images_array


if __name__ == '__main__':     
    ## functions currently have grayscale active. see gray_image variable
    training_img = open_images('data/Train/Salak/0_100.jpg')
    # print(training_img, '\n')
    testing_img = open_images('data/Test/Mango/0_100.jpg')
    # print(testing_img, '\n')

    ## returns all_files in path
    # training_set = get_file_names('data/Train')
    # testing_set = get_file_names('data/Test')

    ## returns all .jpg files
    # jpgs = get_jpgs(training_set)
    # jpgs = get_jpgs(testing_set)
    