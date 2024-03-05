import cv2
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
image_path = "C:/Users/GOKUL/Downloads/images.jpg"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
print(f"Number of eyes detected: {len(eyes)}")
for (x, y, w, h) in eyes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Detected Eyes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
