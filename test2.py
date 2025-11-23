import cv2
import numpy as np
import tensorflow as tf

# ----------------------------
# CONFIGURATION
# ----------------------------
# Path to your trained model
MODEL_PATH = "emotion_cnn_7classes.h5" 

# Path to the photo you want to test
IMAGE_PATH = "/home/harshan/Documents/edl/23BTRCL015/dataset_Face/23BTRCL015-01-SA-01.jpeg"
import numpy as np
import tensorflow as tf

# ----------------------------
# CONFIGURATION
# ----------------------------
# Path to your trained model
MODEL_PATH = "emotion_cnn_7classes.h5" 

# Path to the photo you want to test
IMAGE_PATH = "/home/harshan/Documents/edl/23BTRCL015/dataset_Face/23BTRCL015-01-SA-01.jpeg"  # <--- REPLACE THIS with your specific image filename

# Must match the Training Script!
IMG_SIZE = 96 

# Label mapping
INDEX_TO_NAME = {
    0: "Happy",
    1: "Sad",
    2: "Surprised",
    3: "Angry",
    4: "Fear",
    5: "Neutral",
    6: "Disgust",
}

# ----------------------------
# LOAD RESOURCES
# ----------------------------
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except OSError:
    print(f"Error: Could not find model at {MODEL_PATH}")
    exit()

# Load Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# PREDICT FUNCTION
# ----------------------------
def predict_emotion_on_photo(img_path):
    # 1. Read Image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Detect Faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected in this image.")
        return

    print(f"Found {len(faces)} face(s).")

    # 4. Loop through detected faces
    for (x, y, w, h) in faces:
        # Extract face ROI (Region of Interest)
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize to match training input (96x96)
        roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        
        # Normalize (0-1)
        roi_gray = roi_gray.astype("float32") / 255.0
        
        # Expand dimensions to match model input: (1, 96, 96, 1)
        roi_gray = np.expand_dims(roi_gray, axis=0) 
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict
        prediction = model.predict(roi_gray, verbose=0)
        max_index = int(np.argmax(prediction))
        confidence = prediction[0][max_index] * 100
        predicted_emotion = INDEX_TO_NAME[max_index]

        # Draw Rectangle & Label
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        text = f"{predicted_emotion} ({confidence:.1f}%)"
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 5. Show Result
    cv2.imshow("Emotion Detection - Press any key to exit", img)
    cv2.waitKey(0) # Waits indefinitely for a key press
    cv2.destroyAllWindows()

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    predict_emotion_on_photo(IMAGE_PATH)  # <--- REPLACE THIS with your specific image filename

# Must match the Training Script!
IMG_SIZE = 96 

# Label mapping
INDEX_TO_NAME = {
    0: "Happy",
    1: "Sad",
    2: "Surprised",
    3: "Angry",
    4: "Fear",
    5: "Neutral",
    6: "Disgust",
}

# ----------------------------
# LOAD RESOURCES
# ----------------------------
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except OSError:
    print(f"Error: Could not find model at {MODEL_PATH}")
    exit()

# Load Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# PREDICT FUNCTION
# ----------------------------
def predict_emotion_on_photo(img_path):
    # 1. Read Image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Detect Faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected in this image.")
        return

    print(f"Found {len(faces)} face(s).")

    # 4. Loop through detected faces
    for (x, y, w, h) in faces:
        # Extract face ROI (Region of Interest)
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize to match training input (96x96)
        roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        
        # Normalize (0-1)
        roi_gray = roi_gray.astype("float32") / 255.0
        
        # Expand dimensions to match model input: (1, 96, 96, 1)
        roi_gray = np.expand_dims(roi_gray, axis=0) 
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict
        prediction = model.predict(roi_gray, verbose=0)
        max_index = int(np.argmax(prediction))
        confidence = prediction[0][max_index] * 100
        predicted_emotion = INDEX_TO_NAME[max_index]

        # Draw Rectangle & Label
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        text = f"{predicted_emotion} ({confidence:.1f}%)"
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 5. Show Result
    cv2.imshow("Emotion Detection - Press any key to exit", img)
    cv2.waitKey(0) # Waits indefinitely for a key press
    cv2.destroyAllWindows()

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    predict_emotion_on_photo(IMAGE_PATH)