import cv2 as cv
import numpy as np
path1=r"D:\SEM 6\Comp_vision\Programs\Sample_Images\WaterMark_Sample2.png"
path2=r"D:\SEM 6\Comp_vision\Programs\Sample_Images\Sample9.jpg"
img1=cv.imread(path1)
img2=cv.imread(path2)
gray=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
resize_img1=cv.resize(img1,(780,540))
resize_img2=cv.resize(img2,(780,540))
not_operator=cv.bitwise_not(resize_img1)
add=cv.addWeighted(resize_img2,1.0,not_operator,6.9,0)
cv.imwrite("Output23_Original_Image.jpg",img2)
cv.imwrite("Output23_WaterMark.jpg",img1)
cv.imwrite("Output23_Final_Image.jpg",add)
