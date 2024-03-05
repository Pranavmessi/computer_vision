import cv2

def main():
    image_path = "C:/Users/GOKUL/Downloads/fall-autumn-red-season.jpg"
    image = cv2.imread(image_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    fg_mask = bg_subtractor.apply(image)
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)
    fg = cv2.bitwise_and(image, image, mask=fg_mask)

    cv2.imshow('Original Image', image)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Foreground', fg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
