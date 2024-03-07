import cv2
cap = cv2.VideoCapture("D:\SEM 6\Comp_vision\Programs\Sample_Images\Moving_car.mp4")
while(True):
	ret, frame = cap.read()
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame,'MOVING CAR SAMPLE',(50, 50),font, 1,(0, 255, 255),2,cv2.LINE_4)
	cv2.imshow('Output36_Text_On_Vid.jpg', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
