import cv2
face_cascade = cv2.CascadeClassifier(r'D:\Computer_Vision\Xml_File\haarcascade-frontalface-default.xml')
img = cv2.imread(r'D:\Computer_Vision\Sample_Images\Homelander.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imwrite('Face_Detection.jpg', img)
cv2.waitKey()
