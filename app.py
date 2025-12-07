import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

# ----------------------------
# CONFIGURATION
# ----------------------------
app = Flask(__name__)

# Folders for saving images
UPLOAD_FOLDER = 'static/uploads'
PREDICT_FOLDER = 'static/predictions'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICT_FOLDER'] = PREDICT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

IMG_SIZE = 96
INDEX_TO_NAME = {
    0: "Happy", 1: "Sad", 2: "Surprised", 3: "Angry", 
    4: "Fear", 5: "Neutral", 6: "Disgust"
}

# ----------------------------
# LOAD MODEL & RESOURCES
# ----------------------------
print("Loading model and cascade...")
model = tf.keras.models.load_model("emotion_mobilenet_v2.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("Resources loaded.")

# ----------------------------
# ROUTES
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        # 1. Save Original Image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 2. Process Image (The Logic)
        img = cv2.imread(filepath)
        
        # Convert to Gray for Face Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        prediction_text = "No faces detected."
        
        if len(faces) > 0:
            prediction_text = f"Detected {len(faces)} face(s)."
            
            for (x, y, w, h) in faces:
                # Extract Face
                roi_color = img[y:y+h, x:x+w]
                
                # Preprocess for Model
                roi_resized = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB) # Model expects RGB
                roi_norm = roi_rgb.astype("float32") / 255.0
                roi_input = np.expand_dims(roi_norm, axis=0)

                # Predict
                prediction = model.predict(roi_input, verbose=0)
                max_index = int(np.argmax(prediction))
                confidence = prediction[0][max_index] * 100
                emotion_label = INDEX_TO_NAME[max_index]

                # Draw Box (Green)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw Text
                text = f"{emotion_label} ({confidence:.1f}%)"
                text_y = y - 10 if y - 10 > 10 else y + h + 20
                cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 3. Save Processed Image
        output_filename = "pred_" + filename
        output_path = os.path.join(app.config['PREDICT_FOLDER'], output_filename)
        cv2.imwrite(output_path, img)

        # 4. Return Template with Image Paths
        return render_template('index.html', 
                               uploaded_image=filepath,
                               result_image=output_path,
                               message=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)