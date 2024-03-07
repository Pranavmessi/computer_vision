import cv2
import os
cam = cv2.VideoCapture(r"D:\SEM 6\Comp_vision\Programs\Sample_Images\Moving_car.mp4")
frameno = 0
while(True):
   ret,frame = cam.read()
   if ret:
      name = str(frameno) + '.jpg'
      print ('new frame captured...' + name)

      cv2.imwrite(name, frame)
      frameno += 1
   else:
      break
cam.release()
cv2.destroyAllWindows()
