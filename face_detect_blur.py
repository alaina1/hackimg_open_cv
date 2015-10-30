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
    resultImage = image.copy()
    imageName = imagePath.split('.')[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    if len(faces) == 0:
        wrongCount=wrongCount+1


    #print "Found {0} faces!".format(len(faces))


    # Draw a rectangle around the faces
    faceNum = 0
    for (x, y, w, h) in faces:
        #get the rectangle img around all the faces
        #cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,0), 5)
        subFace = image[y:y+h, x:x+w]
        # apply a gaussian blur on this new recangle image
        subFace = cv2.GaussianBlur(subFace,(29, 29), 30)
        # merge this blurry rectangle to our final image
        resultImage[y:y+subFace.shape[0], x:x+subFace.shape[1]] = subFace
        faceNum = faceNum + 1

    faceFileName = "images/blurred_faces_in_image/" + imageName  + "_face_" + str(faceNum)  + ".jpg"
    cv2.imwrite(faceFileName, resultImage)

end = time.time()
print end-start

files = len(os.listdir(imageDirPath))
percentWrong = float(wrongCount)/float(files)
print "{0:0.2f}".format(percentWrong)
