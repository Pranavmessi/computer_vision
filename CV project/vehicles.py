import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('vehicle.mp4')

# Load the car cascade classifier
car_cascade = cv2.CascadeClassifier('haarcascade_cars.xml')

# Initialize frame1
ret, frame1 = cap.read()

# Function to get centroid
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Minimum contour dimensions
min_contour_width = 40
min_contour_height = 40
offset = 10
line_height = 550
matches = []
cars_detected = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    # Draw rectangles around the detected cars and count them
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        crop_img = frame[y:y+h, x:x+w]

    # Vehicle detection based on motion
    d = cv2.absdiff(frame1, frame)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

        if not contour_valid:
            continue

        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
        cv2.line(frame, (0, line_height), (1200, line_height), (0, 255, 0), 2)
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
        cx, cy = get_centroid(x, y, w, h)

        for (x, y) in matches:
            if (line_height + offset) > y > (line_height - offset):
                cars_detected += 1
                matches.remove((x, y))
                print(cars_detected)

    cv2.putText(frame, "Total Vehicles Detected: " + str(cars_detected), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)

    cv2.imshow("OUTPUT", frame)

    if cv2.waitKey(1) == 27:
        break

    frame1 = frame

cv2.destroyAllWindows()
cap.release()
