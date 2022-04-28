import cv2 as cv

capture = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    # Reading frame  
    _, frame = capture.read()

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detecting  faces  
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    # print(f'Number of faces found = {len(faces_rect)}')

    # Draw rectangle around each face  
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    # Display
    cv.imshow('Video', frame)

    # Stop if escape key is pressed 
    k = cv.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
capture.release()