import cv2
video_path = "C:/Users/GOKUL/Downloads/video (2160p).mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image_path = "C:/Users/GOKUL/Downloads/fall-autumn-red-season.jpg"
    cv2.imwrite(image_path, frame)
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
