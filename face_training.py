import cv2,os
har_file = 'haarcascade_frontalface_default.xml'
dataset = 'Image_dataset'
sub_dataset = 'Jaii'

path = os.path.join(dataset,sub_dataset)
if not os.path.isdir(path):
    os.makedirs(path)
(width,height) = (130,100)

face_cascade = cv2.CascadeClassifier(har_file)

webcam = cv2.VideoCapture(0)

count = 1
while count < 51:
    _,img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(width,height))
        cv2.imwrite('%s/%s.jpg'%(path,count),face_resize)
        count += 1
    cv2.imshow('Face',img)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
