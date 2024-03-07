import numpy as np  
import cv2  
img = cv2.imread(r"D:\SEM 6\Comp_vision\Programs\Sample_Images\Pagani.jpg",1)  
cv2.ellipse(img, (250, 150), (80, 20), 5, 0, 360, (0,0,255), -1)  
cv2.imwrite('Output34_Drawing_Ellipse.jpg',img)  
cv2.waitKey(0)  
cv2.destroyAllWindows() 
