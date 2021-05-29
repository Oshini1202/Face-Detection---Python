import cv2
from random import randrange

# Lording pretrained data
trained_face_data = cv2.CascadeClassifier('data\haarcascade_frontalface_default.xml')

# Extracting the image
webcam = cv2.VideoCapture(0)

#Frames Detection
while True:
    #Read the current fame
    successful_frame_read , frame = webcam.read()

    ## turning the video to gray scale(back and white)
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_cordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # drawing rectacangle
    for (x, y, w, h) in face_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow("Face Detection App", frame)
    key = cv2.waitKey(1)

    #stop the code when Q is pressed
    if key == 81 or key == 113:
        break

webcam.release()

