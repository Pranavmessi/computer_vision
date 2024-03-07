import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)
print(kernel)
path = "D:\SEM 6\Comp_vision\Programs\Sample_Images\Sample1.jpg"
img =cv2.imread(path)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite("Output1.jpg",imgGray)
cv2.waitKey(0)
