import cv2
import numpy as np

def apply_filter(image, filter_type):
    if filter_type == 'blur':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'edges':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    else:
        return image

def main():
    image_path = "C:/Users/GOKUL/Downloads/fall-autumn-red-season.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found.")
        return

    filter_type = input("Choose a filter (blur/edges/sharpen/none): ")

    filtered_image = apply_filter(image, filter_type)

    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
