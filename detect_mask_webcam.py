# Import necessary packages for webcam access and model loading
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Load the trained model from disk
print("[INFO] loading model...")
model = load_model("mask_detector.model.h5")  # Load the model from the specified path

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Open the default camera (0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()  # Capture a frame
    if not ret:  # Check if the frame was captured successfully
        print("[INFO] failed to grab frame")
        break  # Exit the loop if frame capture failed

    # Resize the frame to 224x224 pixels for model input
    image = cv2.resize(frame, (224, 224))  # Resize the frame
    image = img_to_array(image)  # Convert the frame to a NumPy array
    image = preprocess_input(image)  # Preprocess the image for MobileNetV2
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input shape

    # Classify the input image
    predictions = model.predict(image)  # Make predictions on the resized frame
    mask, withoutMask = predictions[0]  # Get the probabilities for each class
    label = "Mask" if mask > withoutMask else "No Mask"  # Determine the label based on probabilities
    confidence = max(mask, withoutMask) * 100  # Calculate confidence percentage

    # Display the label and probability on the frame
    label_text = f"{label}: {confidence:.2f}%"  # Create label text
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Draw text on the frame

    # Show the output frame
    cv2.imshow("Webcam", frame)  # Display the frame in a window

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'q' key press
        break  # Exit the loop if 'q' is pressed

# Release the webcam and close windows
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
