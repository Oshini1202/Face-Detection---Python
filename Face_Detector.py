import cv2
from random import randrange

# Lording pretrained data
trained_face_data = cv2.CascadeClassifier('data\haarcascade_frontalface_default.xml')

# Extracting the image
img = cv2.imread("images\sjk1.jpg")

# turning the image to gray scale(back and white)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
face_cordinates = trained_face_data.detectMultiScale(grayscaled_img)

#drawing rectacangle
for (x, y, w, h) in face_cordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

#display the image
cv2.imshow("Face Detection App", img)
cv2.waitKey()


print("hi code is working")
