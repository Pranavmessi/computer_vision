import cv2
face_cascade = cv2.CascadeClassifier(r"D:\SEM 6\Comp_vision\Programs\Xml_Files\haarcascade-frontalface-default.xml")
image = cv2.imread(r"D:\SEM 6\Comp_vision\Programs\Sample_Images\Homelander.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
print(f"Number of faces detected: {len(faces)}")
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
cv2.imwrite('Output28_FaceCount.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
