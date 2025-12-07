import cv2
import numpy as np
import tensorflow as tf

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_PATH = "emotion_mobilenet_v2.h5"
IMG_SIZE = 96

# Map indices to labels (Must match your training)
INDEX_TO_NAME = {
    0: "Happy", 1: "Sad", 2: "Surprised", 3: "Angry", 
    4: "Fear", 5: "Neutral", 6: "Disgust"
}

# ----------------------------
# LOAD RESOURCES
# ----------------------------
print("Loading model... please wait.")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except OSError:
    print(f"Error: Could not find model at {MODEL_PATH}")
    exit()

# Load Face Detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# START WEBCAM
# ----------------------------
# '0' is usually the default built-in webcam. Try '1' if you have an external cam.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting video stream... Press 'q' to quit.")

while True:
    # 1. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Optional: Flip frame mirror-style
    frame = cv2.flip(frame, 1)

    # 2. Convert to Grayscale for Face Detection (faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Detect Faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.3, 
        minNeighbors=5, 
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # 4. Extract Face ROI (Region of Interest)
        # Note: We cut from 'frame' (color), not 'gray'
        roi_color = frame[y:y+h, x:x+w]

        # ------------------------------------------------
        # PREPROCESSING (Must match training exactly!)
        # ------------------------------------------------
        try:
            # Resize to 96x96
            roi_processed = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
            
            # Convert BGR (OpenCV standard) to RGB (Model standard)
            roi_processed = cv2.cvtColor(roi_processed, cv2.COLOR_BGR2RGB)
            
            # Normalize (0 to 1)
            roi_processed = roi_processed.astype("float32") / 255.0
            
            # Expand dims to become (1, 96, 96, 3)
            roi_processed = np.expand_dims(roi_processed, axis=0)

            # 5. Predict
            prediction = model.predict(roi_processed, verbose=0)
            max_index = int(np.argmax(prediction))
            confidence = prediction[0][max_index] * 100
            predicted_emotion = INDEX_TO_NAME[max_index]

            # ------------------------------------------------
            # DRAWING UI
            # ------------------------------------------------
            # Color: Green for high confidence, Red for low
            color = (0, 255, 0) if confidence > 50 else (0, 0, 255)
            
            # Draw Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw Text Background (for readability)
            label = f"{predicted_emotion} ({confidence:.0f}%)"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - 35), (x + text_w, y), color, -1)
            
            # Draw Text
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"Error processing face: {e}")

    # 6. Show Result
    cv2.imshow('Live Emotion Detector', frame)

    # 7. Quit Logic (Press 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()