import sys
import cv2
import os
import time

# Get user supplied values

start = time.time()

imageDirPath = sys.argv[1]
cascPath = sys.argv[2]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
wrongCount = 0

# Read the images
for imagePath in os.listdir(imageDirPath):
    image = cv2.imread(imageDirPath + imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    if len(faces) > 0:
        wrongCount=wrongCount+1



    #print "Found {0} faces!".format(len(faces))


    # Draw a rectangle around the faces
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

percentWrong = float(wrongCount)/(len(os.listdir(imageDirPath)))
print "Percentage of incorrect classifications: {0}".format(percentWrong*100)
end = time.time()
print "Time to run: {0}".format(end-start)
