import cv2
import time
import numpy as np
from glasses_detector import GlassesClassifier, GlassesDetector

classifier = GlassesClassifier()#glasses classifier
detector = GlassesDetector()
cap = cv2.VideoCapture(0) #0 for laptop, 1 for phone

while True:
    start = time.time()
    ret, frame = cap.read()

    #send cv2 in rgb, not bgr
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #classifier checks for glasses. Prediction is true or false
    prediction = classifier(image=frame_rgb, format="bool") 

    #time and fps calculation
    end = time.time()
    second = end - start
    fps = 1 / second if second > 0 else 0

    #prints pfs and glasses status 
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = f"FPS: {fps:.2f} Glasses: {'Yes' if prediction else 'No'}"
    cv2.putText(frame, txt, (200, 440), font, 1, (255, 255, 255), 2, cv2.LINE_4)

    cv2.imshow('Glasses Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
