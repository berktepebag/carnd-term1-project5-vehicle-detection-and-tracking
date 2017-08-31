
## UDACITY Self Driving Car Nano Degree Term 1 Project 5

**Vehicle Detection Project**

System:

Intel Q9400 @3200
6 gb DDR2 @800mhz ram
NVIDIA gtx 1070 8 gb
Tensorflow GPU version on Anaconda
OpenCV

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.JPG "Car - Not Car Example"
[image2]: ./output_images/car-notcar-hog.JPG "Car - Not Car HOG Example "
[image3]: ./output_images/HLS_Scale_1_5.JPG "HLS Scale 1.5"
[image4]: ./output_images/HLS_Scale_2.JPG "HLS Scale 2"
[image5]: ./output_images/HSV_Scale_1_5.JPG "HSV Scale 1.5"
[image6]: ./output_images/HSV_Scale_1_5_Tree_Shadow.JPG "HSV Scale 1.5 Shadow Tree Problem"
[image7]: ./output_images/HSV_Scale_2.JPG "HSV Scale 2"
[image8]: ./output_images/HSV_Scale_2_Tree_Shadow.JPG "HSV Scale 2 Shadow Tree Problem"
[image9]: ./output_images/YCrCb_Scale_1_5.JPG "YCrCb Scale 1.5"
[image10]: ./output_images/YCrCb_Scale_2.JPG "YCrCb Scale 2"
[image11]: ./output_images/YCrCb_Scale_1_6_1.JPG "YCrCb Scale 1.6"
[image12]: ./output_images/YCrCb_Scale_1_6_2.JPG "YCrCb Scale 1.6"
[image13]: ./output_images/YCrCb_Scale_1_6_3.JPG "YCrCb Scale 1.6"
[image14]: ./output_images/YCrCb_Scale_1_6_1_threshold.JPG "YCrCb Scale 1.6 Threshold"
[image15]: ./output_images/YCrCb_Scale_1_6_2_threshold.JPG "YCrCb Scale 1.6 Threshold"
[image16]: ./output_images/YCrCb_Scale_1_6_3_threshold.JPG "YCrCb Scale 1.6 Threshold"

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

##Pipeline of Detecting Vehicles

###Histogram of Oriented Gradients (HOG)

####1. In the first cell I imported all the necessary libraries. At the second cell, imported car and notcar images from vehicle and non-vehicle data sets. Here is an example of car and notcar images.

![alt text][image1]

In the third cell defined convert_color() to find out which color space fits best with the example images. Defined get_hog_features() using skimage.hog() and bin_spatial(), color_hist() definitions which we will feed into extract_features() to obtain hog features from it.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

In the fourth cell defined extract_features(), which takes in image(s) with HOG parameters and returns HOG features. single_img_features() (defined in fifth cell) is same with extract_features() except it takes only one image and is used to create example results. slide_window() takes in an image and divides into windows and returns a window_list. draw_boxes() takes in image and the result of search_window() which returns positive detections to draw identified cars as rectangles on raw image. 

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters pre threshold:

#### Applied HLS, retured too many false-positives for both Scale: 1.5 and 2
![alt text][image3]
![alt text][image4]
#### Applied HSV, it was good with cars but tree shadows cause too many false-positives
![alt text][image5]
#### HSV Creates too much False Positives due to Tree Shadow 
![alt text][image6]
![alt text][image7]
![alt text][image8]
#### Applied YCrCb, Scale: 1.5 found cars but detected number was as low as 1 or none. Scale: 2 could not found anything at all. 
![alt text][image9]
![alt text][image10]


#Define Paramaters
y_start_stop =[400,700] : Cuts Image into half where the cars are not visible anymore. 
ystart=y_start_stop[0]	: Size of cut image in upper y direction
ystop=y_start_stop[1]	: Size of cut image in lower y direction
xy_window = (96,96)		: Size of the searching windows, chosen according to try outs where both cars can be found most of the time.
overlap = 0.5			: Overlap ratio of the searching boxes 
color_space='YCrCb'		: Color Space, Tried LUV, YUV, HLS, HSV as can be seen at images. YCrCb suited the best, especially cause less problems with tree shadows. 
spatial_size=(32, 32)	: resize the car - notcar images into 32x32 pixels where image is still recognazible and provides enough info to get HOG features of the images
hist_bins=32			: 
orient=9				: Gradient samples divided into # bins, since image is not big 6-9 gives sufficient information
pix_per_cell=8			: Dividing search windows into #**2 cells
cell_per_block=2		: Pixels moved with each step  
hog_channel="ALL"		: Picked all color channels since there is no dominating color
spatial_feat=True		: Use spatial features
hist_feat=True			: Use Color Features
hog_feat=True			: Use HOG features


####3. Training a classifier using selected HOG features and color features

I trained a linear SVM using sklearn.svm library at eight cell. Tried different color features, accuracies was:
RGB Acc: 0.94, HLS Acc: 0.98, HSV Acc: 0.99, YCrCb Acc: 0.9935. Picked YCrCb as color features since accuracy was the highest.

###Sliding Window Search

####1.Implementing a sliding window search, deciding scales to search and how much to overlap windows

Sliding window size determined with try-outs. I picked a size where close and far cars can be found. Sticked with the overlapping % given in the lecture. 


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image11]
![alt text][image12]
![alt text][image13]

---

### Video Implementation

Here's a [link to my video result](https://drive.google.com/open?id=0B1qa2SOuBDHOMk9YUG9mLVVBN00)


####2. Filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap (images on the right) from all nine frames and bounding boxes are drawn (on the left):
![alt text][image14]
![alt text][image15]
![alt text][image16]
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It took some time to decide the parameters, since vehicle sizes are changing during the video it is not possible to find an exact seach window size. It can be fixed by using Multi-Scale windows. After deciding a point sufficient to draw rectangles on the detected cars, rectangles were changing size rapidly which was not eye pleasing. Added draw_labeled_bboxes() an bbox list which saves the bbox's found and when finds a new bbox combines n-1th box with new found bbox with the ratio of 0.8(new bbox) to 0.2(n-1th bbox). Adding more bbox's may cause smoother results.

Here is a video [With Smoothing](https://drive.google.com/open?id=0B1qa2SOuBDHObk9KemlfQUJGQW8)
and [Without Smoothing](https://drive.google.com/open?id=0B1qa2SOuBDHObGl1UEt4d2tsVGc)

