import cv2
import imutils

cascade = 'Models\cars.xml'
car_cascade = cv2.CascadeClassifier(cascade)

cam = cv2.VideoCapture(0)
while True:
    img = cam.read()[1]
    img = imutils.resize(img, width=1000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 6)
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),2)
    cv2.imshow('Car Detection', img)
    n = str(len(cars))
    q = cv2.waitKey(10)
    if q == ord('q'):
        break
print('Number of cars detected: ', n)
cam.release()
cv2.destroyAllWindows
