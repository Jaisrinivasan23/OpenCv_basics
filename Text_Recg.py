from PIL import Image
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

def recText(filename):
    text = pytesseract.image_to_string(Image.open(filename))
    return text

info = recText('Datasets\Screenshot 2024-04-15 184535.png')
print(info)