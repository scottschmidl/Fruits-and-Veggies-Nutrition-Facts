import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.cluster import KMeans  # unsupervised learning
from skimage.filters import gaussian

## working with an image
apple = io.imread('data/Train/Apple/AB0_100.jpg')
# print(apple, '*******')
# print(type(apple))
# print(apple.shape)

io.imshow(apple)
# plt.show()

## 0 in [40:50, 40:50, 0] is for the red channel, change to 1 for green 
## or 2 for blue, leave empty and delete comma for all the channels,
##  then add a comma and 3 to reshape(10,10) 
apple_middle = apple[40:50, 40:50].reshape(10, 10, 3)
# print(apple_middle)
io.imshow(apple_middle)
# plt.show()

## pixel intensities
apple_middle_red = apple_middle[:, :, 0]
apple_middle_green = apple_middle[:, :, 1]
apple_middle_blue = apple_middle[:, :, 2]
# print(apple_middle_red, 'red intensity****\n')
# print(apple_middle_green, 'green_intensity******\n')
# print(apple_middle_blue, 'blue_intensity******\n')

## what if we don't care about color in an image?
apple_gray = rgb2gray(apple)
# print("Image shapes:")
# print("Sunset RGB (3 channel): ", apple.shape)
# print("Sunset (gray): ", apple_gray.shape)

# print("\nMinimum and maximum pixel intensities:")
# print("Original sunset RGB: ", apple.min(), ",", apple.max())
# print("Sunset gray (grayscale):", apple_gray.min(), ",", apple_gray.max())
io.imshow(apple_gray)
# plt.show()

apple_gray_values = np.ravel(apple_gray)
apple_gray_values.shape

## plot common featurization approach for images
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.hist(apple_gray_values, bins=256)
ax.set_xlabel('pixel intensities', fontsize=14)
ax.set_ylabel('frequency in image', fontsize=14)
ax.set_title("Apple Grayscale image histogram", fontsize=16)
# plt.show()

## plot threshold intensity
apple_threshold_intensity = 0.9 # play with this
setting_sun = (apple_gray >= apple_threshold_intensity).astype(int)
io.imshow(setting_sun, cmap='gray')
# plt.show()

## time to learn about image. 
## the (-1, 3) means give me 3 columns and however many rows into order to make
## the 3 columns
print(apple.reshape(-1,3).mean(axis=0)) # the average r,g,b value

X = apple.reshape(-1,3)
clf = KMeans(n_clusters=3).fit(X)
# print(clf.cluster_centers_)
# print(clf.labels_.shape)
labels = set(clf.labels_)
# print(labels)

## color here means nothing, only labels
# plt.imshow( clf.labels_.reshape(100, 100))
# plt.show()

## Featurization-Looking for edges in images
sobel_img = filters.sobel(apple_gray)
io.imshow(sobel_img[20:30, 0:10])
# plt.show()
sobel_img[20:30, 0:10].round(2)
io.imshow(gaussian(apple_gray, sigma=3))
# plt.show()
