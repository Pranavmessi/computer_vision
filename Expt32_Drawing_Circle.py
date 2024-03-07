import numpy as np  
import cv2  
img = cv2.imread(r"D:\SEM 6\Comp_vision\Programs\Sample_Images\Nostalgia.jpg",1)  
cv2.circle(img,(450,450), 355, (0,255,0), -1)  
cv2.imwrite('Output32_Drawing_Circle.jpg',img)  
cv2.waitKey(0)  
cv2.destroyAllWindows() 
