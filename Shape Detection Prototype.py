# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:20:38 2024

@author: kvmal
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour

img_resize = (800, 800)

img = cv2.imread("Oval.jpg")
img = cv2.resize(img, img_resize, interpolation=cv2.INTER_LINEAR)


edgesx = cv2.Sobel(img, -1, dx=1, dy=0, ksize=1)
edgesy = cv2.Sobel(img, -1, dx=0, dy=1, ksize=1)
edges = edgesx + edgesy

sobel_img = edges
laplace_img = cv2.Laplacian(img, cv2.CV_64F)

s = np.linspace(0, 2*np.pi, 400)
r = (img_resize[0]/2) + (img_resize[0]/2)*np.sin(s)
c = (img_resize[1]/2) + (img_resize[1]/2)*np.cos(s)
init = np.array([r, c]).T

snake = active_contour(gaussian(img, 3, preserve_range=False),
                       init, alpha=0.015, beta=10, gamma=0.001)

snake_converted = snake.astype("float32")

area = cv2.contourArea(snake_converted)
perimeter = cv2.arcLength(snake_converted, True)

circularity = (4 * np.pi * area) / (perimeter * perimeter)

threshold = 0.02

if (abs(circularity - 1) > threshold):
    title = "Oval"
else:
    title = "Circle"

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.title(title)
plt.show()