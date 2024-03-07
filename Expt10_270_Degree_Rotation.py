import cv2
def rotate_image(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    rotation_angle = 270
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    cv2.imwrite('Output10/Rotate-270.jpg', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    image_path = "D:\SEM 6\Comp_vision\Programs\Sample_Images\Sample10.jpg"
    rotate_image(image_path)
