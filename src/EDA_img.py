import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.cluster import KMeans  # unsupervised learning
from skimage.filters import gaussian
import os, numpy, PIL #avg image
from PIL import Image #avg image
# from image_loader import open_images

### start: part of initial EDA
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

# def get_jpgs(t_set):
#     images_array = []
#     for path in t_set:
#         if path[-3:] == 'jpg':
#             images_array.append(open_images(path, fruit_name))
#     return images_array
### end: part of initial EDA

def see_fruit(fruit_name):
    fruit_path = os.listdir('data/fruits_vegetables/{}'.format(fruit_name))
    imlist_fruit = [filename for filename in fruit_path if filename[-3:] in "jpg"]
    # print(imlist_fruit[:2], 'imlist********\n')    
    for img in imlist_fruit:
        # print(img, '***img***')
        fruit_img = io.imread('data/fruits_vegetables/{}/{}'.format(fruit_name, img))
        # print(fruit_img[0])
        # print(fruit_name, '*******')
        # print(type(fruit_img))
        # print(fruit_img.shape)
        # io.imshow(fruit)
        # plt.show()
        ## 0 in [40:50, 40:50, 0] is for the red channel, change to 1 for green 
        ## or 2 for blue, leave empty and delete comma for all the channels,
        ## then add a comma and 3 to reshape(10,10) 
        # fruit_name_middle = fruit_img[40:50, 40:50].reshape(10, 10, 3)
        # print(fruit_name_middle)
        # io.imshow(fruit_name_middle)
        # plt.show()    
        fruit_color_values = np.ravel(fruit_img)
    # print(fruit_img, '*********')
    return imlist_fruit, fruit_img, fruit_color_values

### start avg img
## Access all jpg files in directory
def avg_img(fruit_name):
    imlist_fruit = see_fruit(fruit_name)[0]
    # print(imlist_fruit, '\n')    
    #below just looks at size of first img in imlistfruit
    w, h = Image.open('data/fruits_vegetables/{}/{}'.format(fruit_name, imlist_fruit[0])).size 
    # print(w, h)
    N=len(imlist_fruit)
    # print(N, '********')
    arr=numpy.zeros((h, w, 3), numpy.float)
    ## Build up average pixel intensities, casting each image as an array of floats
    for im in imlist_fruit:
        imarr = numpy.array(Image.open('data/fruits_vegetables/{}/{}'.format(fruit_name, im)), dtype=numpy.float)
        arr=arr+imarr/N        
    ## Round values in array and cast as 8-bit integer
    arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)
    # print(arr, '*********')
    ## Generate, save and preview final image
    out = Image.fromarray(arr, mode="RGB")
    # out.save("images/tomato_average.jpg")
    # out.show()    
    return out
### end avg image

def scale_fruit(fruit_name):
    fruit_img = see_fruit(fruit_name)[1]    
    # fruit_name_middle_red = fruit_name_middle[:, :, 0]
    # fruit_name_middle_green = fruit_name_middle[:, :, 1]
    # fruit_name_middle_blue = fruit_name_middle[:, :, 2]
    # print(fruit_name_middle_red, 'red intensity****\n')
    # print(fruit_name_middle_green, 'green_intensity******\n')
    # print(fruit_name_middle_blue, 'blue_intensity******\n')
    fruit_name_gray = rgb2gray(fruit_img)    
    # print("Image shapes:")
    # print("Fruit RGB (3 channel): ", fruit_name.shape)
    # print("Fruit (gray): ", fruit_name_gray.shape)
    # print("\nMinimum and maximum pixel intensities:")
    # print("Original sunset RGB: ", fruit_name.min(), ",", fruit_name.max())
    # print("Sunset gray (grayscale):", fruit_name_gray.min(), ",", fruit_name_gray.max())
    io.imshow(fruit_name_gray)
    # io.imsave('images/tomato_avg_image_grayscale.png', fruit_name_gray)
    # plt.title('Tomato Average Image Grayscale')
    # plt.show()    
    fruit_name_gray_values = np.ravel(fruit_name_gray)
    # print(fruit_name_gray_values.shape)
    return fruit_name_gray, fruit_name_gray_values

def plot_pixel_intensities(fruit_name):
    # fruit_color_values = see_fruit(fruit_name)[2] #this plots as color
    fruit_name_gray_values = scale_fruit(fruit_name)[1] #this plots as gray    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.hist(fruit_name_gray_values, bins=256)
    ax.set_xlabel('Pixel Intensities', fontsize=14)
    ax.set_ylabel('Frequency In Image', fontsize=14)
    # ax.set_title("{} Grayscale Image Histogram".format(fruit_name), fontsize=16)
    # plt.savefig('images/AVG_{}_grayscale_pixel_intensities.png'.format(fruit_name))
    # plt.show()
    return 'cookie monster'

def plot_threshold_intensity(fruit_name, fruit_location):
    fruit_img = see_fruit(fruit_name)[1]
    fruit_name_gray = scale_fruit(fruit_name)[0]    
    fruit_name_threshold_intensity = 0.7 # play with this
    fruit_tensity = (fruit_name_gray >= fruit_name_threshold_intensity).astype(int)
    io.imshow(fruit_tensity, cmap='gray')
    plt.title('Grayscale {} Threshold Intensity'.format(fruit_name))
    plt.savefig('images/avg_{}_Grayscale_threshold.png'.format(fruit_name))
    plt.show()
    ## time to learn about image. 
    ## the (-1, 3) means give me 3 columns and however many rows into order to make
    ## the 3 columns
    # print(fruit_name.reshape(-1,3).mean(axis=0)) '''the average r,g,b value'''
    X = fruit_img.reshape(-1,3)
    clf = KMeans(n_clusters=3).fit(X)
    # print(clf.cluster_centers_)
    # print(clf.labels_.shape)
    # labels = set(clf.labels_)
    # print(labels)
    # color here means nothing, only labels
    plt.imshow(clf.labels_.reshape(100, 100))
    # plt.title('KMeans Cluster For Avg Color {}'.format(fruit_name))
    # plt.savefig('images/avg_{}_color_kmeans_clusters.png'.format(fruit_name))
    # plt.show()
    return 'cookie monster'

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
    ## returns all_files in path
    # tomato_set = get_file_names('data/fruits_vegetables/Tomato')
    # pear_set = get_file_names('data/fruits_vegetables/Pear')

    ## returns all .jpg files
    # jpgs = get_jpgs(tomato_set)
    # jpgs = get_jpgs(pear_set)

    ## look at fruit
    see = see_fruit('Tomato')
    # see = see_fruit('Pear')

    ## get average images
    # plot_avg_images = avg_img('Tomato')
    # plot_avg_images = avg_img('Pear')

    # ## look at scale
    # color_gray = scale_fruit('Tomato')
    # color_gray = scale_fruit('Pear')

    ## plot common featurization approach for images. pixel intensities  
    # plot_feat = plot_pixel_intensities('Tomato')
    # plot_feat = plot_pixel_intensities('Pear')
      

    ## plot threshold pixel intensity
    # plot_intensity = plot_threshold_intensity('Tomato')
    # plot_intensity = plot_threshold_intensity('Pear')

    ## Featurization-Looking for edges in images
    # plot_edges = look_at_edges('Tomato')
    # plot_edges = look_at_edges('Pear')
