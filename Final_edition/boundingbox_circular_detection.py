import cv2
import time
import numpy as np
import torch
import os
from src.glasses_detector import GlassesClassifier, GlassesDetector  # Import from the local repo

# Initialize the classifier and detector with the correct weights and settings
# Remove task and use only the arguments supported by GlassesClassifier
# Initialize the classifier
#import torch
#from glasses_detector import GlassesClassifier

# Path to your checkpoint file
checkpoint_path = "checkpoints/classification-circularglasses-small-epoch=01-val_loss=1.263.ckpt"

# Initialize the GlassesClassifier
classifier = GlassesClassifier(kind="circularglasses", size="small", weights=False)  # Disable automatic weight loading

# Load the full checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Extract the state_dict (actual model weights)
model_state_dict = checkpoint["state_dict"]

# Get the target model's state_dict keys for reference
target_state_dict = classifier.model.state_dict()

# Adjust and filter the loaded state_dict
adjusted_state_dict = {
    key.replace("model.", ""): value
    for key, value in model_state_dict.items()
    if key.replace("model.", "") in target_state_dict
}

# Load the filtered state_dict into the classifier's model
classifier.model.load_state_dict(adjusted_state_dict)
# Use the classifier as intended in your video processing pipeline

# classifier = GlassesClassifier(
#     kind="circularglasses",  # Focus on circular glasses
#     size="small",           # Use the small model variant
#     weights="checkpoints/classification-circularglasses-small-epoch=01-val_loss=1.263.ckpt"  # Path to weights
# )


detector = GlassesDetector()  # Initialize detector (update if specific weights are needed)

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
    # prediction = classifier.predict(image=frame_rgb, format="bool")
    probability = classifier.predict(image=frame_rgb, format="proba")  # Returns the probability of circular glasses
    prediction = probability > 0.5  # Threshold to decide presence of glasses

    if prediction:
        with torch.inference_mode():
            # Predict bounding boxes for the frame
            boxes = detector.predict(image=frame_rgb, format="int")

            # Draw bounding boxes on the frame
            frame_rgb = detector.draw_boxes(
                image=frame_rgb,
                boxes=boxes,
                colors="red",  # Box color
                fill=False,    # Boxes are outlines, not filled
                width=3        # Box line width
            )

            if boxes:
                x, y, w, h = boxes[0]
                region = frame[y:h, x:w]
                cv2.imwrite('found_glasses.jpg', region)

    # Convert the annotated frame back to BGR for OpenCV display
    frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)

    # Calculate FPS
    end = time.time()
    second = end - start
    fps = 1 / second if second > 0 else 0

    # Display FPS and glasses detection status on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = f"FPS: {fps:.2f}, Prec: {probability:.2f} Circularglasses: {'Yes' if prediction else 'No'}"
    cv2.putText(frame_bgr, txt, (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Glasses Detection', frame_bgr)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Search image when 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        path = os.path.abspath("found_glasses.jpg")
        print(f"Saved glass region to {path}")

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
