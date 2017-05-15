**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/image1.png
[image2]: ./images/image2.png
[image3]: ./images/image3.png
[image4]: ./images/image4.png
[image5]: ./images/image5.png
[project.ipynb]: https://github.com/grdshch/CarND-Vehicle-Detection/blob/master/project.ipynb

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

All code is in [project.ipynb] Jupyter notebook.

[project.html](http://htmlpreview.github.io/?https://github.com/grdshch/CarND-Vehicle-Detection/blob/master/project.html) contains the executed result of the Jupyter notebook.

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cell 4 of the [project.ipynb] Jupyter notebook, function `get_hog_features`.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed two images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

I loaded all the `vehicle` and `non-vehicle` images and extracted hog features, spatially binned color and histogram of color features to use them to train the classifier. Also I normalized features using `sklearn.preprocessing.StandardScaler`. Here is the example of original image, histogram of its features and histogram of normalized features:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters , more and less orientations and different number of pixels per cell and cells per block. Results didn't differ much and I selected parameters mentioned before as quite good ones. Also I tried different color spaces and selected `YCrCb` one.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn.svm.LinearSVC` classifier. I tried different classifiers, but couldn't get better score and `LinearSVC` was the fastest one. Also I used `sklearn.model_selection.GridSearchCV` fine tuning classifier's C parameter. I selected value `C=0.005` which gave the best score.

Result score is `98.9%`.

---

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to calculate HOG features for the whole image and then to slide a window along it with different scales. This is much faster then calculating HOG for each window. Here is the example of the grid I got with overlapped windows and with scale `1.5`:

![alt text][image3]

Due to different size of cars on images I selected two scales - `1.25` and `1.5` to get better results. I searched using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector

Sliding window, detecting cars and collecting bounding boxes is done in cell 13 of [project.ipynb] in function `find_cars`. It supports several scales at once and return all found bounding boxes.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some example images:

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [project_video_result.mp4](https://github.com/grdshch/CarND-Vehicle-Detection/blob/master/project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

This is done in cell 19 of [project.ipynb], function `process` which contains the final pipeline to process the project video.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six test images with found bounding boxes, their corresponding heatmaps and result labels:

![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My solution may be too hardcoded to the project video: size of window to search cars (actually, at may be calibrated for the specific self driving car), size of cars - some motorcyvle or huge truck may be missed due to their size.

Heatmap approach unites several cars into one if they are close, this is OK for the project but may confuse self driving car if it will try to detect size of the car, distance to it, etc.

To make the car detection more smart I think it makes sense to add more computer vision algorithms, if we found a car bounding box using heatmap we can then try to find exact car contour. This will separate two close cars, make bounding box better fit for the car and more stable from frame to frame.
