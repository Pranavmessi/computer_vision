import numpy as np  
import cv2  
img = cv2.imread(r"D:\SEM 6\Comp_vision\Programs\Sample_Images\Vecna.jpg",1)  
cv2.rectangle(img,(15,25),(200,150),(0,255,255),15)  
cv2.imwrite('Output33_Drawing_Box.jpg',img)  
cv2.waitKey(0)  
cv2.destroyAllWindows() 
