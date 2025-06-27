# ML--attendance-with-face-recognition
A machine learning-powered attendance system using real-time face recognition. Built with OpenCV, face_recognition, and Python, this project detects and recognizes faces to automatically log attendance into a CSV file. Ideal for classrooms, offices, and restricted-entry systems.

A real-time face recognition-based attendance system using OpenCV, face_recognition, and Python. This project includes:
- Webcam-based attendance marking
- Student image capture and model training
- Automatic CSV-based attendance recording



---

## 🚀 Features
- 🔒 Secure face recognition using HOG model
- ✅ Attendance is marked only once per person per day
- 📸 Simple webcam image capture for training
- 📂 Encoded face data saved using `pickle`
- 📊 CSV file for easy attendance export
- ⚡ Fast and lightweight (no deep learning GPU required)

---

## 📦 Requirements

Install the required packages:

```bash
pip install opencv-python face_recognition pandas numpy
````

> You may also need to install `dlib` dependencies manually based on your system.

---

## 🧠 Step 1: Train the Model

Run the training script to collect images and encode them:

```bash
python train_faces_attendance.py
```

### ✍️ Instructions:

* Enter a **student name** (e.g., Alan, Biju) when prompted.
* Capture up to 5 images per student (or press `q` to stop early).
* When finished adding students, type `w` to exit.
* The script will then encode all captured images and save them into `trained_faces_attendance.pkl`.

---

## 🎯 Step 2: Run the Attendance System

Once training is complete, run the main attendance script:

```bash
python attendance.py
```

### 🎥 Instructions:

* The webcam will open and begin scanning faces.
* If a trained face is detected, it will mark attendance in `attendance.csv`.
* Attendance is recorded once per student per day.
* Press `q` to quit the program.

---

## 📊 Attendance Output

Attendance is saved in `attendance.csv`:

```csv
Name,Date,Time
name1,2025-06-27,08:45:22
name2,2025-06-27,08:47:10
```

---

## 📌 Notes

* Ensure **consistent lighting** while capturing and recognizing faces.
* Avoid blurry or low-resolution images for better accuracy.
* Supports `.jpg`, `.jpeg`, and `.png` formats.

---

## 👨‍💻 Author

**Alan Biju**
BCA student, Mar Augusthinose College
📍 Ramapuram, Kottayam

---

---

## 📷 Preview

> You can optionally add screenshots of:
>
> * The attendance system in action
> * A sample of `attendance.csv`
> * Dataset folder structure

---

## 🙌 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

