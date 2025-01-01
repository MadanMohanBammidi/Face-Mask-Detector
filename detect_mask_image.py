# import the necessary packages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import argparse
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model.h5",
	help="path to trained face mask detector model")
args = vars(ap.parse_args())

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# load the input image, preprocess it, and prepare it for classification
image = load_img(args["image"], target_size=(224, 224))
image = img_to_array(image)
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)

# classify the input image
print("[INFO] classifying image...")
(predictions) = model.predict(image)

# interpret the predictions
mask, withoutMask = predictions[0]
label = "Mask" if mask > withoutMask else "No Mask"
confidence = max(mask, withoutMask) * 100

# display the label and probability
print(f"{label}: {confidence:.2f}%")

