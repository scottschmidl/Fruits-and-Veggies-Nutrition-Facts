from skimage.io import imread
from skimage.color import rgb2gray
import os

def open_training_images():
    train_images = imread(r"data/Training/Apple/AB0_100.jpg")        
    # show_img(image)
    red, yellow =   train_images.copy(), train_images.copy()
    red[:,:,(1,2)] = 0
    yellow[:,:,2]=0
    # show_images(images=[red,yellow], titles=['Red Intensity','Yellow Intensity'])
    # gray_image = rgb2gray(train_images) #for making image gray scale
    # show_images(images=[image,gray_image],titles=["Color","Grayscale"])
    return train_images

def open_test_images(): #NEED TO REARRANGE FOLDERS
    test_images = imread(r"data/Test/Apple/AB3_100.jpg")
    # show_img(image)
    red, yellow =   test_images.copy(), test_images.copy()
    red[:,:,(1,2)] = 0
    yellow[:,:,2]=0
    # show_images(images=[red,yellow], titles=['Red Intensity','Yellow Intensity'])
    # gray_image = rgb2gray(test_images) #for making image gray scale
    # show_images(images=[image,gray_image],titles=["Color","Grayscale"])
    return test_images

# '''
#     For the given path, get the List of all files in the directory tree 
# '''
# # dirName = '../data/json_dump';
# def get_file_names(dirName):
#     # create a list of file and sub directories 
#     # names in the given directory 
#     listOfFile = os.listdir(dirName)
#     allFiles = list()
#     # Iterate over all the entries
#     for entry in listOfFile:
#         # Create full path
#         fullPath = os.path.join(dirName, entry)
#         # If entry is a directory then get the list of files in this directory 
#         if os.path.isdir(fullPath):
#             allFiles = allFiles + get_file_names(fullPath)
#         else:
#             allFiles.append(fullPath)
#     print(allFiles)                
#     return allFiles        
# # dirName = '../data/json_dump';

if __name__ == '__main__':
     
    training_set = open_training_images()
    test_set = open_test_images()
    
    print("Colored image shape: ", training_set.shape)
    #print(training_set)
        
    #print("Grayscale image shape:", gray_image.shape)
    print("Colored image shape: ",test_set.shape)
    #print(test_set)