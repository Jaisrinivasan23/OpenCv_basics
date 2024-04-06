import numpy as np
import imutils
import cv2

# Path to the prototxt file and the model file
protext = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
# Minimum confidence threshold for detections
confTresh = 0.2

# List of classes for the MobileNet SSD model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor", "mobile"]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained model
try:
    net = cv2.dnn.readNetFromCaffe(protext, model)
except cv2.error as e:
    print("Error loading model:", e)
    exit(1)

# Open a video capture object for the webcam
vs = cv2.VideoCapture(0)

# Loop indefinitely over frames from the video stream
while True:
    # Read the next frame from the video stream
    _, frame = vs.read()

    # Resize the frame to have a maximum width of 500 pixels
    frame = imutils.resize(frame, width=500)
    # Get the dimensions of the frame
    (h, w) = frame.shape[:2]

    # Preprocess the frame for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # Perform object detection
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum threshold
        if confidence > confTresh:
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])
            # Calculate the bounding box coordinates of the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Display the label and confidence of the detection
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Display the frame with detections
    cv2.imshow("Frame", frame)
    # Check for the 'q' key to quit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video stream and close all windows
vs.release()
cv2.destroyAllWindows()
