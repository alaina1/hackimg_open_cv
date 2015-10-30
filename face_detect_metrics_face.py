import sys
import cv2
import os
import time

# Get user supplied values

imageDirPath = sys.argv[1]
cascPath = sys.argv[2]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
wrongCount = 0

start = time.time()

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

    if len(faces) == 0:
        wrongCount=wrongCount+1


    #print "Found {0} faces!".format(len(faces))


    # Draw a rectangle around the faces
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

end = time.time()
print "time to run: {0}".format(end-start)

files = len(os.listdir(imageDirPath))
percentWrong = float(wrongCount)/float(files)*100
print "Percentage of incorrect classifications: "
print "{0:0.2f}".format(percentWrong)
