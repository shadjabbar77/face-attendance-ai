# Face Recognition Attendance AI

A Python-based face recognition attendance system that learns known faces, identifies people in new images, draws bounding boxes, and logs recognized names to an attendance CSV file.

## Features

* Detects faces in uploaded images
* Learns known faces from a local `known/` folder
* Saves face encodings for reuse
* Identifies known people in new images
* Draws bounding boxes and names on output images
* Supports adjustable match tolerance
* Logs recognized faces to `attendance.csv`
* Prevents duplicate attendance entries
* Skips unknown faces when saving attendance records

## Technologies

Python, OpenCV, face_recognition, dlib, NumPy, Pillow, Git, VS Code

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python3 train.py
```

Recognize faces:

```bash
python3 recognize.py unknown/unknown_test.jpg 0.55
```

View attendance log locally:

```bash
cat attendance.csv
```

## Privacy Note

This project was tested locally using sample images. Face images, test images, output images, attendance logs, and saved face encodings are not included in this repository for copyright, privacy, and biometric data protection reasons.

To run the project, users can add their own consented images into the `known/` and `unknown/` folders locally.
