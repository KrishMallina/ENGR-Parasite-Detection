from PIL import Image
import cv2
import numpy as np

# Import the image
image = Image.open('Strongyle2.png')

# Test if the image loaded correctly
image = cv2.imread('Strongyle2.png')
if image is None:
    print("Error: Unable to load image.")

# Read the image in grayscale
image = cv2.imread('Strongyle2.png', cv2.IMREAD_GRAYSCALE)

# # Apply a Gaussian blur to reduce noise
# blurredImage = cv2.GaussianBlur(image, (5, 5), 0)

blurredImage = image.copy()
for _ in range(3):  # Apply 5 iterations (you can adjust the number as needed)
    blurredImage = cv2.GaussianBlur(blurredImage, (3, 3), 0)


# Apply Canny edge detection to detect edges
edges = cv2.Canny(blurredImage, 100, 200)

# Apply the Laplacian filter
laplacianImage = cv2.Laplacian(blurredImage, cv2.CV_64F)

# Convert the result back to an 8-bit image
sharpImage = np.uint8(np.clip(laplacianImage, 0, 255))

# Apply thresholding to highlight edges

_, edgeImage = cv2.threshold(sharpImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the original, sharpened, and edges images
cv2.imshow('Original', image)
cv2.imshow('Sharpened', sharpImage)
cv2.imshow("Edges", edgeImage)
cv2.imshow("secondEdges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
