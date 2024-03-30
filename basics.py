import cv2
import imutils

img = cv2.imread('image.jpg') # Read the image

cv2.imshow('image', img) # Display the image

cv2.imwrite('image_copy.jpg', img) # Save the image

cv2.waitKey(0) # Wait for a key press

cv2.destroyAllWindows() # Close all windows

grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale image

gaussBlur = cv2.GaussianBlur(grayimage,(21,21),0) # 

thresholdimg = cv2.threshold(grayimage,150,255,cv2.THRESH_BINARY)[1]

