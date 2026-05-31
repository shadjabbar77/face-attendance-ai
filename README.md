## Privacy Note

This project was tested locally using sample images. Face images, test images, output images, and saved face encodings are not included in this repository for copyright, privacy, and biometric data protection reasons.

To run the project, users can add their own consented images into the `known/` and `unknown/` folders locally.

## Features

- Detects faces in uploaded images
- Learns known faces from a local folder
- Saves face encodings for reuse
- Identifies known people in new images
- Draws bounding boxes and names on output images
- Supports adjustable tolerance for stricter or looser matching
- Logs recognized faces to an attendance CSV file
- Skips unknown faces when saving attendance records
- Adds timestamp, name, match distance, and image filename to the log

## Technologies

Python, OpenCV, face_recognition, dlib, NumPy, Pillow

## How to Run

```bash
python3 train.py
python3 recognize.py unknown/unknown_test.jpg 0.55

Run recognition with adjustable tolerance:

```bash
python3 recognize.py unknown/unknown_test.jpg 0.55
```

View attendance log locally:

```bash
cat attendance.csv
```

`attendance.csv` is excluded from GitHub for privacy.