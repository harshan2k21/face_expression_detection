import cv2
import numpy as np
import tensorflow as tf

# ----------------------------
# CONFIGURATION
# ----------------------------
# Path to your NEW MobileNet model
MODEL_PATH = "emotion_mobilenet_v2.h5" 

# Path to the photo you want to test
IMAGE_PATH = "/home/harshan/Documents/edl/Face_expression_detection/test/dis_02.jpg"  # <--- REPLACE THIS with your specific image filename

IMG_SIZE = 96 

INDEX_TO_NAME = {
    0: "Happy", 1: "Sad", 2: "Surprised", 3: "Angry", 
    4: "Fear", 5: "Neutral", 6: "Disgust"
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

    # --- FIX: RESIZE IF TOO BIG ---
    height, width = img.shape[:2]
    max_height = 800  # Set a maximum height (e.g., 800 pixels)

    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        img = cv2.resize(img, (new_width, max_height))
        print(f"Image resized to {new_width}x{max_height} for display.")
    # ------------------------------

    # 2. Convert to Grayscale ONLY for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Detect Faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected.")
        # Still show the image even if no face is found, so you know it loaded
        cv2.imshow("Emotion Detection", img) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print(f"Found {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        # 4. Extract face from the ORIGINAL COLOR image
        roi_color = img[y:y+h, x:x+w]
        
        # 5. Resize to 96x96
        roi_color = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
        
        # 6. Convert BGR to RGB
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        
        # 7. Normalize
        roi_color = roi_color.astype("float32") / 255.0
        
        # Expand dims: (1, 96, 96, 3)
        roi_color = np.expand_dims(roi_color, axis=0) 

        # 8. Predict
        prediction = model.predict(roi_color, verbose=0)
        max_index = int(np.argmax(prediction))
        confidence = prediction[0][max_index] * 100
        predicted_emotion = INDEX_TO_NAME[max_index]

        # Draw box & text
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{predicted_emotion} ({confidence:.1f}%)"
        
        # Ensure text doesn't go off the top of the image
        text_y = y - 10 if y - 10 > 10 else y + h + 20
        cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 9. Show Result
    cv2.imshow("Emotion Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_emotion_on_photo(IMAGE_PATH)