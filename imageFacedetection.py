import cv2

image = cv2.imread(r"E:\Opencv\Project 1\Images\face.png")
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# convert to grayscale format
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 1.1 is the scale factor and 4 is the minimum neighbors
faces = facecascade.detectMultiScale(gray, 1.1, 4)
        
# Draw bounding boxes around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
cv2.imshow('Output', image)
        


# Release the capture and close windows


cv2.waitKey(0)