import cv2
import face_recognition
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Directory and files
csv_file = "attendance.csv"
model_file = "trained_faces_attendance.pkl"

# Load trained model
try:
    with open(model_file, "rb") as f:
        known_encodings, known_names = pickle.load(f)
except FileNotFoundError:
    print(f"❌ Error: Model file '{model_file}' not found. Run train_faces_attendance.py first.")
    exit()
except Exception as e:
    print(f"❌ Error loading model file: {e}")
    exit()

if not known_encodings:
    print("❌ Error: No face encodings in the model. Train the model with valid images.")
    exit()

# Load or create attendance CSV
try:
    attendance_df = pd.read_csv(csv_file)
except FileNotFoundError:
    attendance_df = pd.DataFrame(columns=["Name", "Date", "Time"])
    attendance_df.to_csv(csv_file, index=False)

# Function to mark attendance
def mark_attendance(name):
    global attendance_df
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Avoid duplicate for the same date
    if not ((attendance_df["Name"] == name) & (attendance_df["Date"] == date_str)).any():
        new_row = {"Name": name, "Date": date_str, "Time": time_str}
        attendance_df = pd.concat([attendance_df, pd.DataFrame([new_row])], ignore_index=True)
        attendance_df.to_csv(csv_file, index=False)
        print(f"✅ {name} marked present at {time_str} on {date_str}")
        return True
    else:
        print(f"ℹ️ {name} already marked present today.")
        return False

# Start webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Error: Could not open webcam. Try a different camera index (e.g., 1).")
    exit()

print("\n🎥 Face Recognition Attendance System")
print(f"Recognizing faces using model: {model_file}")
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cam.read()
    if not ret or frame is None or frame.size == 0:
        print("❌ Error: Failed to capture video frame.")
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert to RGB and ensure uint8
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1], dtype=np.uint8)

    # Detect faces using HOG model for speed
    try:
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        if not face_locations:
            cv2.imshow("📸 Attendance System - Press 'q' to Quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Compute face encodings
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    except Exception as e:
        print(f"⚠️ Error processing frame: {e}")
        continue

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if any(matches):
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]
                mark_attendance(name)

        # Draw box and label
        top, right, bottom, left = [v * 4 for v in face_location]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # Show the live feed
    cv2.imshow("📸 Attendance System - Press 'q' to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

print(f"\n📁 Attendance session ended. Records saved in {csv_file}.")