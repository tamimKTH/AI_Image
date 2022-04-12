# pillow
#
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# img = Image.open("test_pic-p.jpg")
#print(type(img))
#img.show()
#print(img.format)

# convert img to ndarray type
# img1 = np.asarray(img)
#print(type(img1))

####################################

# Matplotlib -> pyplot

# img3 = mpimg.imread("test_pic-p.jpg")
#print(type(img3))
#print(img3.shape)
#plt.imshow(img3)
#plt.colorbar()


############################

# psych
# pip install scikit-image
# for image segmentation, geometric transformation, color space manipluation, analysis, filtering,
# feature detection(extract bunch of features for machine learning for machine learning (traditional ML like random forest or support vector))
# only needs couple of lines of code within psych for machine learning

from skimage import io, img_as_float, img_as_ubyte

# image = io.imread("test_pic-p.jpg") # .astype(np.float)
# image_float = img_as_float(image)
#print(image_float)
#plt.imshow(image)

#####################################

# opencv
# pip install opencv-python
# lbrary of programming functions that is good for computer vision, images, videos, live videos
# facial detection, object detection, motion tracking, OCR (Optical Character Recognition), reading numbers, letters from handwriting, public signs
# also good for segmentation and for including artificial neural networks and also deep learning
# has the tools for advanced image proccessing
# Pic color format HSV


import cv2

# image_gray = cv2.imread("test_pic-p.jpg", 0) # gray image
## CV2 handles images as BGR (blue, gray, red) not RGB (Red Green Blue)
# image_BGR = cv2.imread("test_pic-p.jpg", 1) # BGR color image
# image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
# print(type(image_cv))
# cv2.imshow("Gray Image", image_gray)
# cv2.imshow("BGR Color Image", image_BGR)
# cv2.imshow("RGB Color Image", image_RGB)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###########################################

# pip install czifile
# CZI images 5d images


###########################################
# OME-TIF
# pip install apeer-ometiff-library


###########################################

# read multi images not only one (Automation)
# glob from UNIX system allow us look at subdirectory and extract the files names

# pip install glob
import os
import glob
from tkinter import filedialog

dir1 = filedialog.askdirectory()
os.chdir(dir1) # change work directory to navigated one in dir1
print(dir1)
path = "*"
image_gray = cv2.imread("test_pic-p.jpg", 0) # gray image

for file in glob.glob(path):
    print(file)
    a = cv2.imread(file)
    print(a)
    c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    cv2.imshow("color rgb", a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
