import cv2
import face_recognition
import os
import pickle
from datetime import datetime

# Directory and model file
dataset_dir = "dataset"
model_file = "trained_faces_attendance.pkl"

def capture_images_for_student(name, max_images=5):
    student_dir = os.path.join(dataset_dir, name)
    os.makedirs(student_dir, exist_ok=True)

    # Initialize webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Error: Could not open webcam.")
        return False

    count = 0
    print(f"\n📸 Capturing up to {max_images} images for {name}.")
    print(f"Images will be saved in: {student_dir}")
    print("Face will be captured automatically when detected. Press 'q' to stop.\n")

    while count < max_images:
        ret, frame = cam.read()
        if not ret:
            print("❌ Error: Failed to capture image.")
            break

        # Convert to RGB for face detection
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        # Display the capture window
        cv2.imshow("Capture Window", frame)

        # Check for 'q' key press before processing face detection
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"🛑 Stopped capturing for {name}.")
            break

        if face_locations:
            # Save image with timestamp to avoid overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = os.path.join(student_dir, f"{name}_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"✅ Saved: {image_path} ({count + 1}/{max_images})")
            count += 1
            # Add delay to avoid capturing too many similar frames
            cv2.waitKey(500)

    # Release resources
    cam.release()
    cv2.destroyAllWindows()

    if count == 0:
        print(f"⚠️ No images captured for {name}.")
        return False
    print(f"✅ Captured {count} image(s) for {name}.")
    return True

def encode_faces():
    print("\n🔄 Encoding all images...")
    known_encodings = []
    known_names = []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    for student in os.listdir(dataset_dir):
        student_folder = os.path.join(dataset_dir, student)
        if not os.path.isdir(student_folder):
            continue

        print(f"Processing folder: {student_folder}")
        for file in os.listdir(student_folder):
            if not file.lower().endswith(valid_extensions):
                continue

            image_path = os.path.join(student_folder, file)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image, model="hog")
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(student)
                    print(f"✅ Encoded face in {image_path}")
                else:
                    print(f"⚠️ Warning: No face detected in {image_path}")
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")

    if known_encodings:
        with open(model_file, "wb") as f:
            pickle.dump((known_encodings, known_names), f)
        print(f"\n✅ Training complete. Encoded data saved to {model_file} with {len(known_encodings)} faces.")
    else:
        print("\n❌ No valid face encodings found. Model not saved.")

# Main training loop
os.makedirs(dataset_dir, exist_ok=True)
print("🎥 Face Recognition Training System")
print(f"Images will be stored in folder-wise structure under: {dataset_dir}")

while True:
    # Validate and get student name
    name = input("\nEnter student name (e.g., Alan, Anvin, Biju) or press 'w' to finish: ").strip()
    if name.lower() == 'w':
        print("🛑 Exiting capture phase.")
        break
    if not name or not name.isalnum():
        print("❌ Error: Name must be non-empty and contain only letters/numbers.")
        continue

    # Capture images for the student
    if capture_images_for_student(name):
        print(f"✅ Finished capturing images for {name}.")
    else:
        print(f"⚠️ Skipped {name} due to no images captured.")

# Encode faces and save model
encode_faces()