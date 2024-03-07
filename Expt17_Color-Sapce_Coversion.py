import cv2 as cv
sample=cv.imread(r"D:\SEM 6\Comp_vision\Programs\Sample_Images\Samplex.jpg")
sample2=cv.imread("D:\SEM 6\Comp_vision\Programs\Sample_Images\Samplex.jpg")
image=cv.cvtColor(sample,cv.COLOR_BGR2GRAY)
image2=cv.cvtColor(sample2,cv.COLOR_BGR2HSV)
cv.imwrite("Output17_Original.jpg",sample)
cv.imwrite("Output17_BGR2GRAY.jpg",image)
cv.imwrite("Output17_BGR2HSV.jpg",image2)
