from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import cv2
import numpy as np
import torch
from glasses_detector import GlassesClassifier, GlassesDetector
from selenium import webdriver

# Path to ChromeDriver
driver_path = "/Users/katla/Desktop/glassesdetector/chromedriver-mac-x64/chromedriver"

# Initialize the classifier and detector
classifier = GlassesClassifier()  # Glasses classifier
detector = GlassesDetector()      # Glasses detector

# Function to handle cookie popup
def dismiss_cookie_popup(driver):
    try:
        print("Checking for cookie consent popup...")
        # Wait for any potential cookie popup dialog
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@role, 'dialog')]"))
        )
        print("Cookie consent popup detected!")

        # Try clicking "Reject all" or similar buttons
        buttons = driver.find_elements(By.XPATH, "//button")
        for button in buttons:
            button_text = button.text.strip().lower()
            if "reject" in button_text or "decline" in button_text or "accept all" in button_text:
                print(f"Clicking button: {button_text}")
                button.click()
                WebDriverWait(driver, 5).until_not(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@role, 'dialog')]"))
                )
                print("Cookie consent popup dismissed.")
                return True

        print("No suitable button found to dismiss cookie popup.")
    except Exception as e:
        print(f"No cookie consent popup detected or failed to dismiss: {e}")
    return False

# Function to perform reverse image search
def reverse_image_search(image_path):
    options = Options()
    # Uncomment the line below for headless mode (optional)
    # options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)

    try:
        print("Opening Google Images...")
        driver.get("https://images.google.com")

        # Handle cookie popup
        dismiss_cookie_popup(driver)

        # Wait for the camera icon to load and be clickable
        print("Waiting for the camera icon to become clickable...")
        camera_icon = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@aria-label='Search by image']"))
        )
        print("Clicking the camera icon...")
        driver.execute_script("arguments[0].click();", camera_icon)

        # Wait for 'Upload a file' option to load
        print("Waiting for 'Upload a file' link to appear...")
        upload_tab = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'upload a file')]"))
        )
        print("Clicking 'Upload a file' link...")
        upload_tab.click()

        # Wait for the file input to appear
        print("Waiting for the file input field...")
        file_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.NAME, "encoded_image"))
        )
        print("Uploading image...")
        file_input.send_keys(image_path)

        print("Waiting for search results to load...")
        time.sleep(5)

        # Get and print the current URL (search results page)
        search_results_url = driver.current_url
        print("Google Search Results URL:", search_results_url)

    except Exception as e:
        print(f"An error occurred during the reverse image search: {e}")

    finally:
        print("Browser will remain open for inspection.")
        input("Press Enter to close the browser...")
        driver.quit()

# Open the video capture (0 for laptop webcam, or 1 for external camera)
cap = cv2.VideoCapture(0)

while True:
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
        print("Glasses detected!")
        with torch.inference_mode():
            # Predict bounding boxes for the frame
            boxes = detector.predict(image=frame_rgb, format="int")

            # Draw bounding boxes on the frame
            frame_rgb = detector.draw_boxes(
                image=frame_rgb,
                boxes=boxes,
                width=3        # Box line width
            )
            
            x, y, w, h = boxes[0]
            # Crop the detected glasses region
            region = frame[y:h, x:w]
            saved_image_path = '/Users/katla/Desktop/glassesdetector/found_glasses.jpg'
            cv2.imwrite(saved_image_path, region)

            print(f"Saved detected glasses region as {saved_image_path}")

            # Perform reverse image search
            reverse_image_search(saved_image_path)
            break  # Exit after finding and searching for glasses

    # Display FPS and glasses detection status on the frame
    cv2.imshow('Glasses Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
