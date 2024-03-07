import cv2
path ="D:\SEM 6\Comp_vision\Programs\Sample_Images\Sample9.jpg"
src = cv2.imread(path)
image = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite("Output9-rotate-90.jpg", image)
cv2.waitKey(0)
