
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
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.JPG
[image3]: ./output_images/HLS_Scale_1_5.JPG
[image4]: ./output_images/HLS_Scale_2.JPG
[image5]: ./output_images/HSV_Scale_1_5.JPG
[image6]: ./output_images/HSV_Scale_1_5_Tree_Shadow.JPG
[image7]: ./output_images/HSV_Scale_2.JPG
[image8]: ./output_images/HSV_Scale_2_Tree_Shadow.JPG
[image9]: ./output_images/YCrCb_Scale_1_5.JPG
[image10]: ./output_images/YCrCb_Scale_2.JPG
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

##Pipeline of Detecting Vehicles

###Histogram of Oriented Gradients (HOG)

####1. In the first cell import all the necessary libraries. At the second cell, imported cars and notcars images from vehicle and non-vehicle data sets. Here is an example of car and notcar images.

![alt text][image1]

In the third cell defined convert_color to find out which color space fits best with the example images. Defined get_hog_features() using skimage.hog() and bin_spatial(), color_hist() definitions which we will feed into extract_features() to obtain hog features from it.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

In the fourth cell defined extract_features(), which takes in image(s) with HOG parameters and returns HOG features. single_img_features() (defined in fifth cell) is same with extract_features() except it takes only one image and is used to create example results. slide_window() takes in an image and divides into windows and returns a window_list. draw_boxes() takes in image and the result of search_window() which returns positive detections to draw identified cars as rectangles on raw image. 

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters pre threshold:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

