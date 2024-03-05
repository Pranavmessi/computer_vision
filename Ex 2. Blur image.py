import cv2
import numpy as np
path = "C:/Users/GOKUL/Downloads/fall-autumn-red-season.jpg"
img =cv2.imread(path)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
cv2.imshow(imgBlur)
cv2.waitKey(0)

