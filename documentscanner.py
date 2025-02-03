import cv2 
import numpy as np

image = cv2.imread(r"E:\Opencv\Project 1\Images\documentscanner2.jpg")
width = 510
height = 597


## Define the function to preprocess the image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0) # Apply GaussianBlur to the grayscale image
    edge = cv2.Canny(blur, 250, 300) # Apply Canny edge detection to the image 
    kernel = np.ones((5,5), np.uint8) # Define the kernel for image dilation and erosion 
    image_dilation = cv2.dilate(edge, kernel, iterations=2) # Apply dilation to the image # dilation is used to increase the white region in the image
    image_erosion = cv2.erode(image_dilation, kernel, iterations=1) # Apply erosion to the image # erosion is used to decrease the white region in the image
    
    return image_erosion


## Define the function to draw the contours on the image
## Define the function to draw the contours on the image
def draw_contours(image):
    contours, hirarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Find the contours in the image 
    maxarea = 0
    biggest = np.array([])
    for cnt in contours:
        area = cv2.contourArea(cnt) # Calculate the area of the contour
        print("Area of the contour: ", area)
        if area > 5000:
            peri = cv2.arcLength(cnt, True) # Calculate the perimeter of the contour
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True) # Approximate the polygonal curve of the contour
            if area > maxarea and len(approx) == 4:
                biggest = approx
    print("Biggest Contour or coner points: ", biggest)   
    print("Biggest Contour Shape: ", biggest.shape) 
    cv2.drawContours(image_contours, biggest, -1, (255, 0, 0), 3)
    return biggest

## Reshape the image to get the scanned document

def reoder(mypoints):
    mypoints = mypoints.reshape((4,2))
    mypoints_new = np.zeros((4,1,2), np.int32)
    
    add = mypoints.sum(1)
    print("Add: ", add)
    
    mypoints_new[0] = mypoints[np.argmin(add)]
    mypoints_new[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints, axis=1)
    mypoints_new[1] = mypoints[np.argmin(diff)]
    mypoints_new[2] = mypoints[np.argmax(diff)]
    
    print("New Points: ", mypoints_new)
    
    return mypoints_new
    


## Define wrap perspective function to get the scanned document

def wrap_perspective(image, biggest):
    biggest = reoder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_output = cv2.warpPerspective(image, matrix, (width, height))
    # to crop the image
    image_cropped = image_output[20:image_output.shape[0]-10, 20:image_output.shape[1]-10]
    image_cropped = cv2.resize(image_cropped, (width, height))
    
    return image_cropped 


## To resize the image
image_resized = cv2.resize(image, (500, 650)) # width, height of the image  # 500, 650

# Preprocess the image
preprocessed_image = preprocess_image(image)
image_contours = image.copy() # Initialize image_contours
biggest = draw_contours(preprocessed_image) # Draw contours on the preprocessed image

# Wrap the perspective to get the scanned document
scanned_document = wrap_perspective(image, biggest)

print("Scanned Image Shape: ", scanned_document.shape)
# Display the images
cv2.imshow("Original Image", image)
cv2.imshow("Preprocessed Image", preprocessed_image)
# cv2.imshow("Image with Contours", biggest) # Display the image with contours``
cv2.imshow("Scanned Document", scanned_document)
cv2.waitKey(0)
cv2.destroyAllWindows()

