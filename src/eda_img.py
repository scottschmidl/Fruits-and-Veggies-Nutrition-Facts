from skimage import io, color, filters, transform
from skimage.filters import gaussian
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

class ImgEDA:

    def see_fruit(fruit_name):
        '''gets images, looks at them, then ravels'''
        fruit_path = os.listdir('data/fruits_vegetables/{}'.format(fruit_name))
        imlist_fruit = [filename for filename in fruit_path if filename[-3:] in ["jpg", "png"]]
        num_100_image = imlist_fruit[100]
        # for img in imlist_fruit: ## <----- use for every image
        fruit_img = io.imread('data/fruits_vegetables/{}/{}'.format(fruit_name, num_100_image))
        ## 0 in [40:50, 40:50, 0] is for the red channel, change to 1 for green
        ## or 2 for blue, leave empty and delete comma for all the channels,
        ## then add a comma and 3 to reshape(10,10)
        if fruit_name == 'Tomato':
            fruit_name_middle = fruit_img[190:200, 200:210].reshape(10, 10, 3)
        else:
            fruit_name_middle = fruit_img[120:130, 150:160].reshape(10, 10, 3)
        io.imshow(fruit_name_middle)
        plt.show()
        fruit_color_values = np.ravel(fruit_img)
        return imlist_fruit, fruit_img, fruit_color_values

    ### start avg img
    def avg_img(fruit_name):
        '''looks at average of image'''
        imlist_fruit = ImgEDA.see_fruit(fruit_name)[0]
        ## below just looks at size of first img in imlist_fruit
        # w, h = Image.open('data/fruits_vegetables/{}/{}'.format(fruit_name, imlist_fruit[0])).size
        N = len(imlist_fruit)
        arr = np.zeros((100, 100, 3), np.float)
        # print(arr.shape)
        ## Build up average pixel intensities, casting each image as an array of floats
        for img in imlist_fruit:
            im = Image.open('data/fruits_vegetables/{}/{}'.format(fruit_name, img))
            im_size = np.resize(im, (100, 100, 3))
            x = np.asarray(im_size, dtype=np.float)
            arr = arr + x/N
        ## Round values in array and cast as 8-bit integer
        arr = np.array(np.round(arr),dtype=np.uint8)
        ## Generate, save and preview final image
        out = Image.fromarray(arr, mode="RGB")
        out.save("images/{}_average_color.jpg".format(fruit_name))
        out.show()
        out_ravel = np.ravel(out)
        return out, out_ravel
    ### end avg image

    def scale_fruit(fruit_name):
        '''makes grayscale and ravels'''
        fruit_img = ImgEDA.see_fruit(fruit_name)[1]
        fruit_name_middle = fruit_img[30:40, 30:40]
        fruit_name_middle_red = fruit_name_middle[:, :, 0]
        fruit_name_middle_green = fruit_name_middle[:, :, 1]
        fruit_name_middle_blue = fruit_name_middle[:, :, 2]
        print(fruit_name_middle_red, 'red intensity****\n')
        print(fruit_name_middle_green, 'green_intensity******\n')
        print(fruit_name_middle_blue, 'blue_intensity******\n')
        fruit_name_gray = rgb2gray(fruit_img)
        print("Image shapes:")
        print("Fruit RGB (3 channel): ", fruit_img.shape)
        print("Fruit (gray): ", fruit_name_gray.shape)
        print("\nMinimum and maximum pixel intensities:")
        print("Original Fruit RGB: ", fruit_img.min(), ",", fruit_img.max())
        print("Fruit gray (grayscale):", fruit_name_gray.min(), ",", fruit_name_gray.max())
        io.imshow(fruit_name_gray)
        io.imsave('images/{}_image_grayscale.png'.format(fruit_name), fruit_name_gray)
        plt.title('{} Image Grayscale'.format(fruit_name))
        plt.show()
        fruit_name_gray_values = np.ravel(fruit_name_gray)
        return fruit_name_gray, fruit_name_gray_values

    def plot_pixel_intensities(fruit_name, grayscale, thres, threshold):
        '''plots pixel (threshold) intensities'''
        if grayscale:
            fruit_name_gray, fruit_gray_color_values = ImgEDA.scale_fruit(fruit_name) #this plots as gray
            if thres and grayscale:
                fruit_tensity = (fruit_name_gray >= threshold).astype(int)
                fig = plt.figure(figsize=(6,6))
                ax = fig.add_subplot(111)
                ax.hist(fruit_gray_color_values, bins=256)
                ax.set_xlabel('Pixel Intensities', fontsize=14)
                ax.set_ylabel('Frequency In Image', fontsize=14)
                io.imshow(fruit_tensity, cmap='gray')
                plt.title('Grayscale {} Threshold Intensity'.format(fruit_name))
                plt.savefig('images/{}_Grayscale_threshold.png'.format(fruit_name))
                plt.show()
            else:
                fig = plt.figure(figsize=(6,6))
                ax = fig.add_subplot(111)
                ax.hist(fruit_gray_color_values, bins=256)
                ax.set_xlabel('Pixel Intensities', fontsize=14)
                ax.set_ylabel('Frequency In Image', fontsize=14)
                ax.set_title("{} Grayscale Image Histogram".format(fruit_name), fontsize=16)
                plt.savefig('images/{}_grayscale_pixel_intensities.png'.format(fruit_name))
                plt.show()
        else:
            fruit_gray_color_values = ImgEDA.see_fruit(fruit_name)[2]
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            ax.hist(fruit_gray_color_values, bins=256)
            ax.set_xlabel('Pixel Intensities', fontsize=14)
            ax.set_ylabel('Frequency In Image', fontsize=14)
            ax.set_title("{} Color Image Histogram".format(fruit_name), fontsize=16)
            plt.savefig('images/{}_color_pixel_intensities.png'.format(fruit_name))
            plt.show()
        return plt

    def plot_kmeans(fruit_name, size):
        '''plots kmeans'''
        ## time to learn about image.
        ## the (-1, 3) means give me 3 columns and however many rows into order to make
        ## the 3 columns
        fruit_img = ImgEDA.see_fruit(fruit_name)[1]
        print(fruit_img.reshape(-1,3).mean(axis=0)) #the average r,g,b value
        X = fruit_img.reshape(-1,3)
        clf = KMeans(n_clusters=3).fit(X)
        print(clf.cluster_centers_)
        print(clf.labels_.shape)
        labels = set(clf.labels_)
        print(labels)
        ## color here means nothing, only labels
        io.imshow(clf.labels_.reshape(size))
        plt.title('KMeans Cluster For Color {}'.format(fruit_name))
        plt.savefig('images/{}_color_kmeans_clusters.png'.format(fruit_name))
        plt.show()
        return plt

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

def main():
    grayscale, thres , threshold = False, False, 0.7
    fruits_list = ['Tomato','Pear']
    ime = ImgEDA()

    for fruit in fruits_list:
        ## LOOK AT FRUIT
        see = ime.see_fruit(fruit)
        # print(see)

        ## GET AVERAGE IMAGES
        plot_avg_images = ime.avg_img(fruit)
        # print(plot_avg_images)

        ## LOOK AT SCALE
        color_gray = ime.scale_fruit(fruit)
        # print(color_gray)

        ## PLOT PIXEL INTENSITIES
        ime.plot_pixel_intensities(fruit, grayscale, thres, threshold)

        ## PLOT KMEANS
        dim = ((480, 322), (320,258))
        ime.plot_kmeans('Tomato', dim[0]) if fruit == 'Tomato' else ime.plot_kmeans('Pear', dim[1])

        ## FEATURIZATION-LOOKING FOR EDGES IN IMAGES
        ime.look_at_edges(fruit, grayimage=None)

if __name__ == '__main__':
    main()
