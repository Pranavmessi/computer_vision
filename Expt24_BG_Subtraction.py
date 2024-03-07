import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
segmentor = SelfiSegmentation()
imgOffice = cv2.imread(r"D:\SEM 6\Comp_vision\Programs\Sample_Images\Vecna.jpg")
imgOffice = cv2.resize(imgOffice, (640, 480))
green = (0, 255, 0)
imgNoBg = segmentor.removeBG(imgOffice, green, threshold=0.50)
cv2.imwrite('Output24_Original_Image.jpg',imgOffice)
cv2.imwrite('Output24_BG_Removed_Img.jpg',imgNoBg)
cv2.waitKey(0)
cv2.destroyAllWindows()
