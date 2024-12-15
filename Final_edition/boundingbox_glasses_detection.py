import cv2
import time
import numpy as np
import torch
import os
from glasses_detector import GlassesClassifier, GlassesDetector
from GoogleSearch import Search   #run pip import google-reverse-search   

# Initialize the classifier and detector
classifier = GlassesClassifier()  # Glasses classifier
detector = GlassesDetector()      # Glasses detector

# Open the video capture (0 for laptop webcam, or 1 for external camera)
cap = cv2.VideoCapture(0)

while True:
    start = time.time()
    ret, frame = cap.read()

    # Ensure frame was read correctly
    if not ret:
        print("Error reading frame from the camera.")
        break

    # Convert frame from BGR to RGB for compatibility with glasses_detector
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Classifier checks for glasses presence
    prediction = classifier(image=frame_rgb, format="bool")

    if prediction:
        with torch.inference_mode():
            # Predict bounding boxes for the frame
            boxes = detector.predict(image=frame_rgb,format="int")

            # Draw bounding boxes on the frame
            frame_rgb = detector.draw_boxes(
                image=frame_rgb,
                boxes=boxes,
                #labels=labels,
                #colors="red",  # Box color
                #fill=False,    # Boxes are outlines, not filled
                width=3        # Box line width
            )
            
            x,y,w,h = boxes[0]
            #print(boxes[0])
            region = frame[y:h,x:w]
            cv2.imwrite('found_glasses.jpg',region)

    # Convert the annotated frame back to BGR for OpenCV display
    frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)

    # Calculate FPS
    end = time.time()
    second = end - start
    fps = 1 / second if second > 0 else 0

    # Display FPS and glasses detection status on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = f"FPS: {fps:.2f} Glasses: {'Yes' if prediction else 'No'}"
    cv2.putText(frame_bgr, txt, (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Glasses Detection', frame_bgr)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # Search image when pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        path = os.path.abspath("found_glasses.jpg")
        Output = Search(file_path=path)

        print(Output)

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
