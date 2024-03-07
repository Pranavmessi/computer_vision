import cv2
import matplotlib.pyplot as plt

# Load the input image
input_image = cv2.imread("D:\SEM 6\Comp_vision\Programs\Sample_Images\Sample6.jpg")

# Split the image into its RGB channels
b, g, r = cv2.split(input_image)

# Calculate histograms for each channel
hist_r = cv2.calcHist([input_image], [2], None, [256], [0, 256])
hist_g = cv2.calcHist([input_image], [1], None, [256], [0, 256])
hist_b = cv2.calcHist([input_image], [0], None, [256], [0, 256])


hist_r1 = cv2.calcHist([r], [0], None, [256], [0, 256])
hist_g1 = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_b1 = cv2.calcHist([b], [0], None, [256], [0, 256])
# Plot histograms
plt.figure()
plt.title("RGB Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

plt.plot(hist_r, color="red", label="Red")
plt.plot(hist_g, color="green", label="Green")
plt.plot(hist_b, color="blue", label="Blue")

plt.figure()
plt.title("RGB Histogram1")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

plt.plot(hist_r1, color="red", label="Red")
plt.plot(hist_g1, color="green", label="Green")
plt.plot(hist_b1, color="blue", label="Blue")
plt.legend()
plt.xlim([0, 256])
plt.show()
