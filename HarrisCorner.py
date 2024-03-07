import cv2
import numpy as np
image_path = 'D:\SEM 6\Comp_vision\Programs\Sample_Images\SampleStar.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
corners = cv2.dilate(corners, None)
threshold = 0.01 * corners.max()
corner_image = np.copy(image)
corner_image[corners > threshold] = [0, 255, 0]
cv2.imshow('Original Image', image)
cv2.imshow('Corners Detected', corner_image)
output_path = 'output_corners.jpg'
cv2.imwrite(output_path, corner_image)
cv2.waitKey(0)
cv2.destroyAllWindows()