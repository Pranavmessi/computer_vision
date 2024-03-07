import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread(r"D:\SEM 6\Comp_vision\Programs\test_images\Test1.png")
stretch_img = cv2.resize(image, (550, 250),
			interpolation = cv2.INTER_LINEAR)
half_img=cv2.resize(image,(0,0),fx=0.1,fy=0.1)
bigger_img=cv2.resize(image,(1050,1610))
plt.subplot(2,2,1),plt.imshow(image)
plt.subplot(2,2,2),plt.imshow(stretch_img)
plt.subplot(2,2,3),plt.imshow(half_img)
plt.subplot(2,2,4),plt.imshow(bigger_img)
plt.show()
