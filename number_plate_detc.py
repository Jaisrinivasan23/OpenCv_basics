import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

image = cv2.imread('Datasets\car_plate.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny_edge = cv2.Canny(gray,170,200)

counters,new = cv2.findContours(canny_edge.copy(),cv2)