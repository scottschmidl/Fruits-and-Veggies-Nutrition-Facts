
![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/fv_world_cloud.png)

# Table of Contents
1. [Background and Motivation](#BackGround-and-Motivation)
2. [Questions](#Questions)
3. [Data](#Data)
4. [Closer Look](#Closer-Look)
5. [Visualization](#Visualization)
6. [Conclusion](#Conclusion)
7. [Photo and Data Credits](#Photo-and-Data-Credits)
8. [Extras](#Extras)


# Background and Motivation

I have always been intrigued by how the cells in our bodies interact with each other, from
how we deal with stress to how what we put in our bodies. Several years ago, I came across
a book called ‘The Optimum Nutrition Bible’, but it wasn’t until last year that I was finally
able to start reading it. The book explains how a variety of stimulus to our bodies affects
our nutrition, which affects our health and wellness. It breaks food, drinks, sleep, stress, and
several other things down to what is in them and how those parts affect our bodies in
negative and positive ways.
It is easy to understand how this book caught my eye and that is why it is the motivation for
this capstone.

# Data

I was able to find a data set on kaggle named ‘Fruits 360’. The data set is comprised of
90,483 100x100 ‘.jpgs’ of fruits and vegetables. 90,380 of those pictures are either a fruit or
vegetable and 103 have multiple fruits or vegetables. The data set has 131 classes of fruits
and vegetables as different varieties of the same item were stored as belonging to
different classes. I, also, built my own data set of each item and some of its nutrition facts,
which I placed into a ‘.csv’ and called as a data frame in pandas.
As part of my check of the data set I verified there were no broken nor fraudulent images.

# The Point

How successfully will my Naive Bayes and Convolution Neural Network train and test the images of
Tomatoes and Pears?

# Closer Look

Average Pear Grayscale Pixel Intensities:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/avg_Pear_grayscale_pixel_intensities.png)

Average Pear Color Pixel Intensities:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/avg_Pear_color_pixel_intensities.png)

Average Pear Color KMean Cluster:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/avg_Pear_color_kmeans_clusters.png)

Average Tomato Grayscale Pixel Intensities:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/avg_Tomato_grayscale_pixel_intensities.png)

Average Tomato Color Pixel Intensities:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/avg_Tomato_color_pixel_intensities.png)

Average Tomato Color KMean Cluster:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/avg_Tomato_color_kmeans_clusters.png)

# Visualization

Grayscale Confusion Matrix:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/gray_confusion_matrix.png)

Grayscale ROC Curve:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/gray_roccurve.png)

Classification Matrix For Grayscale Images:
           
|             | Precision | Recall | F1_Score | Support |
|         --: |       --: |    --: |      --: |     --: |
|         Pear|      0.844|   0.807|     0.825|     1734|
|       Tomato|      0.806|   0.844|     0.824|     1650|
|     accuracy|           |        |     0.825|     3384|
|    macro avg|      0.825|   0.825|     0.825|     3384|
| weighted avg|      0.826|   0.825|     0.825|     3384|

Color Confusion Matrix:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/color_confusion_matrix.png)

Color ROC Curve:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/color_roccurve.png)

Classification Matrix For Color Images:

|             | Precision | Recall | F1_Score | Support |
|         --: |       --: |    --: |      --: |     --: |
|         Pear|      0.858|   0.873|     0.865|     1720|
|      Tomato |      0.866|   0.850|     0.858|     1664|
|     accuracy|           |        |     0.862|     3384|
|    macro avg|      0.862|   0.862|     0.862|     3384|
| weighted avg|      0.862|   0.862|     0.862|     3384|

Edge Confusion Matrix:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/edge_confusion_matrix.png)

Edge ROC Curve:

![](/home/leonardo-leads/Documents/galvanize_dsi/Fruits-and-Veggies-Nutrition-Facts/images/edge_roccurve.png)

Classification Matrix For Edges Images:

|             | Precision | Recall | F1_Score | Support |
|         --: |       --: |    --: |      --: |     --: |
|         Pear|      0.818|   0.806|     0.812|     1664|
|       Tomato|      0.815|   0.827|     0.821|     1720|
|     accuracy|           |        |     0.816|     3384|
|    macro avg|      0.817|   0.816|     0.816|     3384|
| weighted avg|      0.817|   0.816|     0.816|     3384|

# Conclusion

There are too many rotations in the data set, leading to a high risk of training on the test data and therefore poor results. 
As a future project I would like to spend more time cleaning up the fruits, vegetables, and their rotations. I would also like to look at using AWS, as the data set was too long to run all pictures with original pixels.
Something I would like to see come out of this project is an image recognition application that takes a picture of the item and returns it's nutrition facts.

# Data Credit

Horea Muresan, Mihai Oltean, Fruit recognition from images using deep learning, Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.
Here is the GitHub: https://github.com/Horea94/Fruit-Images-Dataset

Information for the nutrition facts was compiled from google searches.

# Extras











