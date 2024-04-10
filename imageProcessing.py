from PIL import Image
import cv2
import numpy as np

# Import the image
image = Image.open('Circle.jpg')

# Test if the image loaded correctly
image = cv2.imread('Circle.jpg')
if image is None:
    print("Error: Unable to load image.")
    
# Read the image in grayscale
image = cv2.imread('Circle.jpg', cv2.IMREAD_GRAYSCALE)

# Apply a Gaussian blur to reduce noise
blurredImage = cv2.GaussianBlur(image, (3, 3), 0)

# Apply the Laplacian filter
laplacianImage = cv2.Laplacian(blurredImage, cv2.CV_64F)

# Convert the result back to an 8-bit image
sharpImage = np.uint8(np.clip(laplacianImage, 0, 255))

# Display the original and sharpened images
cv2.imshow('Original', image)
cv2.imshow('Sharpened', sharpImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
