import cv2

alg = "haarcascade_frontalface_default.xml"

haar_cascade_face = cv2.CascadeClassifier(alg)

cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade_face.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow('Face Detection', img)

    key = cv2.waitKey(10)
    if key == 27:
        break
    
cam.release()
cv2.destroyAllWindows()