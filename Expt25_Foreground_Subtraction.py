import numpy as np
import cv2
img = cv2.imread(r"D:\SEM 6\Comp_vision\Programs\Sample_Images\vecna.jpg")
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (150,50,500,470)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,20,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img1 = img*mask2[:,:,np.newaxis]
cv2.imwrite("Output25_Original_image.jpg",img)
cv2.imwrite('Output25_Foreground_Image.jpg',img1)
