import cv2 as cv
import numpy as np
image=cv.imread(r"D:\SEM 6\Comp_vision\Programs\Sample_Images\Sample3.jpg")
gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
img_gauss=cv.GaussianBlur(gray,(3,3),0)
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv.filter2D(img_gauss, -1, kernelx)
img_prewitty = cv.filter2D(img_gauss, -1, kernely)
cv.imwrite("Output15_PrewittX.jpg", img_prewittx)
cv.imwrite("Output15_PrewittY.jpg", img_prewitty)
cv.imwrite("Output15_Prewitt.jpg", img_prewittx + img_prewitty)
cv.waitkey(0)
cv.destroyAllwindows()
