


import cv2
import os
import threading
import numpy as np
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier


# === CONFIG ===
face_db_path = "Images"
model_name = "SFace"
KNOWN_DISTANCE = 20.0  # cm
KNOWN_WIDTH = 15.0     # cm

# === Load Haar Cascade ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Load model ===
DeepFace.build_model(model_name)

# === KNN Variables ===
knn_classifier = None
face_embeddings = []
face_labels = []

# === Webcam Setup ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === State ===
identified_name = "Scanning..."
match_confidence = 0.0
frame_counter = 0
focal_length = None
raw_distance = 0.0
smoothed_distance = 0.0
distance_update_rate = 5
distance_threshold = 40

# === Get face width ===
def get_face_width(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    _, _, w, _ = faces[0]
    return w

# === Load and encode face images ===
def load_face_embeddings(db_path, model_name):
    global face_embeddings, face_labels, knn_classifier

    print("🔍 Loading face embeddings...")
    for img_name in os.listdir(db_path):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(db_path, img_name)
        person_name = os.path.splitext(img_name)[0]  
        try:
            embedding = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)[0][
                "embedding"]
            face_embeddings.append(embedding)
            face_labels.append(person_name)
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    if face_embeddings:
        print("✅ Training KNN classifier...")
        knn_classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        knn_classifier.fit(face_embeddings, face_labels)
        print("✅ KNN trained with", len(face_labels), "samples.")
    else:
        print("⚠️ No face embeddings loaded.")

# === Recognition Thread ===
def identify_face(frame):
    global identified_name, match_confidence
    try:
        embedding_result = DeepFace.represent(img_path=frame, model_name=model_name, enforce_detection=False)
        if embedding_result:
            embedding = embedding_result[0]["embedding"]
            prediction = knn_classifier.predict([embedding])[0]
            distance = knn_classifier.kneighbors([embedding], return_distance=True)[0][0][0]

            identified_name = prediction
            match_confidence = max(0, 100 - distance * 10)  # Approximate confidence scale
        else:
            identified_name = "Unknown"
            match_confidence = 0.0
    except Exception as e:
        print("Error:", e)
        identified_name = "Error"
        match_confidence = 0.0

# === Load embeddings once ===
load_face_embeddings(face_db_path, model_name)

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read from webcam.")
        continue

    # Calibrate focal length once
    if focal_length is None:
        face_width_px = get_face_width(frame)
        if face_width_px:
            focal_length = (face_width_px * KNOWN_DISTANCE) / KNOWN_WIDTH
            print(f"📏 Focal length calibrated: {focal_length:.2f}px")
        continue

    # Distance estimation (every few frames)
    if frame_counter % distance_update_rate == 0:
        face_width_px = get_face_width(frame)
        if face_width_px and face_width_px > distance_threshold:
            raw_distance = (KNOWN_WIDTH * focal_length) / face_width_px
        else:
            raw_distance = 0.0

        # Smooth it
        alpha = 0.3
        if raw_distance > 0:
            smoothed_distance = alpha * raw_distance + (1 - alpha) * smoothed_distance
        else:
            smoothed_distance = 0

    # Face recognition every 90 frames
    if frame_counter % 90 == 0:
        threading.Thread(target=identify_face, args=(frame.copy(),)).start()

    # === Overlay UI ===
    name_display = f"{identified_name} ({match_confidence:.1f}%)" if identified_name not in ["Unknown", "Scanning...", "Error"] else identified_name
    cv2.putText(frame, name_display, (20, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (0, 255, 0) if identified_name != "Unknown" else (0, 0, 255), 3)

    # Distance text
    if smoothed_distance > 0:
        cv2.putText(frame, f"Distance: {smoothed_distance:.1f} cm", (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Distance: Unknown", (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Face Recognition + Distance", frame)
    frame_counter += 1

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()
