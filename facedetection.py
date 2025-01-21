import cv2

image = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = image.read()
    
    if ret:
        # conver to grayscale format
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 1.1 is the scale factor and 4 is the minimum neighbors
        faces = facecascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
        cv2.imshow('Output', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the capture and close windows

image.release()
cv2.destroyAllWindows()