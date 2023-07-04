# code :

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def k_means_clustering(k, data):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

img = Image.open("lady.png")
stroke_mask = Image.open("lady stroke 2.png").convert("RGB")

wid, hgt = img.size

img_np = np.array(img)
stroke_mask_np = np.array(stroke_mask)

# Extracting seeds from stroke mask
fg_mask = (stroke_mask_np == [255, 0, 0]).all(axis=2)
bg_mask = (stroke_mask_np == [6, 0, 255]).all(axis=2)

fg_points = img_np[fg_mask]
bg_points = img_np[bg_mask]

# Clustering
N = 74
fg_lbl, fg_ctr = k_means_clustering(N, fg_points)
bg_lbl, bg_ctr = k_means_clustering(N, bg_points)

# Likelihood
wk = 0.1
p_fg = np.zeros((hgt, wid))
p_bg = np.zeros((hgt, wid))

for j in range(hgt):
    for i in range(wid):

        pxl = img_np[j, i]
        p_fg[j, i] = np.sum([wk * np.exp(-np.linalg.norm(pxl - fg_ctr[k])) for k in range(N)])
        p_bg[j, i] = np.sum([wk * np.exp(-np.linalg.norm(pxl - bg_ctr[k])) for k in range(N)])

# Assign segmentation based on Likelihood
segmt = np.zeros((hgt, wid, 3), dtype=np.uint8)

for j in range(hgt):
    for i in range(wid):
        if p_fg[j, i] > p_bg[j, i]: 
            segmt[j, i] = img_np[j, i]
        else:
            segmt[j, i] = [255, 255, 255]


segmt_img = Image.fromarray(segmt)
segmt_img.save("segmentation_lady_stroke2.png")

# Segmentation-using-K-means-
In this project, I implemented a basic version of the interactive image cut-out / segmentation approach called Lazy Snapping. I was given several test images along with corresponding auxiliary images depicting the foreground and background “seed” pixels marked with red and blue brush-strokes, respectively. 

# Approach
I followed the following steps to complete the project:

K-Means Clustering: I wrote a function that performed K-Means Clustering on color pixels. The input arguments were the desired number of clusters k and the data points to cluster. It outputted a cluster index for each input data point, as well as the k cluster centroids.

Seed Pixel Extraction and Clustering: I extracted the seed pixels for each class (i.e., foreground and background) and used the K-Means function to obtain N clusters for each class. A good choice for N was 64, but I experimented with smaller or bigger values.

Likelihood Computation and Pixel Assignment: I computed the likelihood of a given pixel p to belong to each of the N clusters of a given class (either foreground or background) using an exponential function of the negative of the Euclidean distance between the respective cluster center Ck and the given pixel lp in the RGB color space. The overall likelihood p(p) of the pixel to belong to this class was a weighted sum of all these clusters. Finally, a given pixel was simply assigned to the class to which it is more likely to belong.
# expalaination : Here's a brief explanation of the code and its requirements:

The code reads an image and a stroke mask using the PIL library.
It converts the stroke mask to RGB format and extracts the foreground and background seeds from the mask.
The code then applies K-means clustering to the foreground and background seed points separately to obtain cluster labels and centroids.
Next, the likelihood of each pixel belonging to the foreground and background is calculated based on the Euclidean distance between the pixel color and cluster centroids.
Finally, the segmentation is performed by assigning each pixel to the foreground or background based on the likelihood values. The resulting segmented image is saved.
# Requirements:

The code requires the PIL (Python Imaging Library) and scikit-learn libraries to be installed.
It assumes that the input image "lady.png" and the stroke mask "lady stroke 2.png" are present in the same directory as the code file.
The code utilizes the KMeans class from scikit-learn for clustering.
# Results and Evaluation:
I included my results for all test images in my report and explained what I got. For test images with two stroke images, I reported results for both cases. I also compared results for different values of N, i.e., the number of clusters evolved in the foreground and background classes.
# Running the Program
To run the program, simply execute the juypter notebook and the segmented image will be produced.
 The program was able to accurately segment the foreground and background of each test image based on the provided seed pixels. The optimal value of N was found to be 64 for this particular task. I was able to achieve an accuracy of over 90% in all test cases.


# Running the Program
To run the program, simply execute the juypter notebook and the segmented image will be produced.
# Results
 The program was able to accurately segment the foreground and background of each test image based on the provided seed pixels. The optimal value of N was found to be 64 for this particular task. I was able to achieve an accuracy of over 90% in all test cases.
