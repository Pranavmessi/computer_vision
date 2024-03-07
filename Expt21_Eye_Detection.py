import cv2 as cv
eye_cascade=cv.CascadeClassifier(r"D:\Computer_Vision\Xml_File\haarcascade_eye.xml")
img=cv.imread(r"D:\Computer_Vision\Sample_Images\Billy2.jpeg")
eye=eye_cascade.detectMultiScale(img,1.9,2)
for (x,y,w,h) in eye:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv.imwrite("Output21_Eye_Detection.jpg",img)
