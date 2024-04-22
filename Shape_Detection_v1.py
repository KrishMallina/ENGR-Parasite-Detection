# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:20:38 2024

@author: kvmal
"""

import cv2
import numpy as np
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from skimage import measure, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation import active_contour

# grayscaling
img_resize = (800, 800)
img = cv2.imread("GetImage (2).png")
img4 = cv2.imread("GetImage (2).png")
img5 = cv2.imread("GetImage (2).png")
img3 = cv2.imread("GetImage (2).png")
img = cv2.resize(img, img_resize, interpolation=cv2.INTER_LINEAR)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgplot=plt.imshow(img)
plt.show()

# contrasting + edge-enhancement for snake contour
# edgesx = cv2.Sobel(img, -1, dx=1, dy=0, ksize=1)
# edgesy = cv2.Sobel(img, -1, dx=0, dy=1, ksize=1)
# edges = edgesx + edgesy
laplace_img = cv2.Laplacian(img, cv2.CV_64F)

#actual image processing for active contour 2
#gaussB = cv2.GaussianBlur(img5,(5,5),0)
blurredImage = img5.copy()
for _ in range(3):  # Apply 5 iterations (you can adjust the number as needed)
    blurredImage = cv2.GaussianBlur(blurredImage, (3, 3), 0)
#applies canny filter and laplace filter, before converting it back to int8
edgesB = cv2.Canny(blurredImage, 100, 160)
laplacianImage = cv2.Laplacian(edgesB, cv2.CV_64F)
sharpImage = np.uint8(np.clip(laplacianImage, 0, 255))
#thresholds image
ret, edgeImage = cv2.threshold(sharpImage, 0, 255, cv2.THRESH_BINARY)
#find active contours
contours5, hierarchy5 = cv2.findContours(edgeImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours5))
# area2 = cv2.contourArea(contours5[0])
# perimeter2 = cv2.arcLength(contours5[0], True)
# makes new array for contours of speific length
contours6 = []
x = 0;
#goes through array of contours
while x < len(contours5):
  #if contours are big enough, it adds it to the new array so it doesnt pick up small contours
  if cv2.contourArea(contours5[x])>800 and cv2.arcLength(contours5[x], True)>150:
      print(cv2.contourArea(contours5[x]))
      #print(cv2.arcLength(contours5[x], True))
      contours6.append(contours5[x])
  x = x+1
print(len(contours6))
cv2.drawContours(img5, contours6, -1, (0, 255, 0), 2)
plt.figure(figsize=[10, 10])
plt.imshow(img5)
plt.show()
contours7 = []
threshold = 0.02
x = 0;
while x < len(contours6):
  areax = cv2.contourArea(contours6[x])
  perimeterx = cv2.arcLength(contours6[x], True)
  circularity = 4 * np.pi * areax / perimeterx * perimeterx
  if(abs(circularity - 1)) > threshold:
     contours7.append(contours6[x])
  x = x + 1

cv2.drawContours(img5, contours7, -1, (0, 255, 0), 2)
plt.figure(figsize=[10, 10])
plt.imshow(img5)
plt.show()

# example calculation of metric for distinguish between circle and oval
#circularity = (4 * np.pi * area) / (perimeter * perimeter)

#if (abs(circularity - 1) > threshold):
    #title = "Oval"
#else:
    #title = "Circle"

#img processing for watershed method
# grayscales and apllies canny filter
# gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
# imgplot = plt.imshow(gray, cmap='gray')
# plt.show()
# edgeApply = cv2.Canny(gray, 10, 250)
# contrast = 1.5
#adds contrast and applies threshold
# img3 = cv2.addWeighted(img3, contrast, np.zeros(img3.shape, img3.dtype), 0, 0)
# ret, thresh = cv2.threshold(edgeApply, 20,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# imgplot45 = plt.imshow(thresh)
# plt.show()
# gray_image = img.sum(-1)
#img processing for second active contour
# gray4 = cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY)

# contours2, hierarchy =
#Code for contour-finding algorithm --> problems = too many contours drawn
# contours = measure.find_contours(gray_image, 0.8)
# img_try = slic(gray_image, n_segments=100, compactness=1)
# imgplot = plt.imshow(img_try)
# plt.show()
# img_convert = np.unique(img_try)

#Code for snake contour algorithm --> problems = not robust enough/ doesn't work
# s = np.linspace(0, 2*np.pi, 400)
# r = (img_resize[0]/2) + (img_resize[0]/2)*np.sin(s)
# c = (img_resize[1]/2) + (img_resize[1]/2)*np.cos(s)
# init = np.array([r, c]).T

#snake = active_contour(gaussian(img, 3, preserve_range=False),init, alpha=0.015, beta=10, gamma=0.001)

# snake_converted = snake.astype("float32")
# threshold = 0.02

#code for slic algorithm
#epsilon = 0.1*cv2.arcLength(snake_converted,True)
#approx = cv2.approxPolyDP(snake_converted,epsilon,True)
#imgplot2 = plt.imshow(approx)
#plt.show()

#image segmentation

# noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)
# imgplot46 = plt.imshow(opening)
# plt.show()

# sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
# imgplot47 = plt.imshow(sure_bg)
# plt.show()

# Finding sure foreground area
# dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
# imgplot48 = plt.imshow(unknown)
# plt.show()
# Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# imgplot49 = plt.imshow(markers)
# plt.show()
# Add one to all labels so that sure background is not 0, but 1
# markers = markers+1

# Now, mark the region of unknown with zero
# markers[unknown==255] = 0


#code to plot slic
# centers = np.array([np.mean(np.nonzero(img_try==i),axis=1) for i in img_try])

# vs_right = np.vstack([img_try[:,:-1].ravel(), img_try[:,1:].ravel()])
# vs_below = np.vstack([img_try[:-1,:].ravel(), img_try[1:,:].ravel()])
# bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111)
# plt.imshow(mark_boundaries(img, img_try))
# plt.scatter(centers[:,1],centers[:,0], c='y')
# plt.show()

# Code to find shape features
# area = cv2.contourArea(snake_converted)
# perimeter = cv2.arcLength(snake_converted, True)

#watershed algorithm plotting
# markers = cv2.watershed(img3,markers)
# img3[markers == -1] = [255,0,0]
#m = cv2.convertScaleAbs(markers)
#plt.imshow(m)
#plt.show()
#fig, ax = plt.subplots(figsize=(6, 6))
#ax.imshow(markers, cmap="tab20b")
#ax.axis('off')
#plt.show()
#labels = np.unique(markers)

#tings = []
#for label in labels[2:]:

# Create a binary image in which only the area of the label is in the foreground
#and the rest of the image is in the background
    #target = np.where(markers == label, 255, 0).astype(np.uint8)

  # Perform contour extraction on the created binary image
    #contours, hierarchy = cv2.findContours(
        #target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    #)
    #tings.append(contours[0])

# Draw the outline
#img3 = cv2.drawContours(img3, tings, -1, color=(0, 23, 223), thickness=2)
#plt.imshow(img3)
#plt.show()

# example calculation of metric for distinguish between circle and oval
#circularity = (4 * np.pi * area) / (perimeter * perimeter)

#if (abs(circularity - 1) > threshold):
    #title = "Oval"
#else:
    #title = "Circle"

# drawing the graph to visualize what the final result looks like
#fig, ax = plt.subplots(figsize=(7, 7))
#ax.imshow(img, cmap=plt.cm.gray)
#ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
#ax.plot(img_try[:, 1], img_try[:, 0], '-b', lw=3)
#ax.set_xticks([]), ax.set_yticks([])
#ax.axis([0, img.shape[1], img.shape[0], 0])

#plt.title(title)
#plt.show()

# drawing the graph to visualize what the final result looks like
#fig, ax = plt.subplots(figsize=(7, 7))
#ax.imshow(img, cmap=plt.cm.gray)
#ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
#ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
#ax.set_xticks([]), ax.set_yticks([])
#ax.axis([0, img.shape[1], img.shape[0], 0])

#plt.title(title)
#plt.show()
