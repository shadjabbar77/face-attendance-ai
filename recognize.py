import sys
import pickle
from pathlib import Path

import cv2
import face_recognition

TOLERANCE = 0.55
ENCODINGS_FILE = "encodings.pkl"
OUTPUT_DIR = Path("output")

if len(sys.argv) < 2:
    print("Usage: python recognize.py unknown/test.jpg")
    sys.exit(1)

image_path = Path(sys.argv[1])

if len(sys.argv) >= 3:
    TOLERANCE = float(sys.argv[2])

print(f"Using tolerance: {TOLERANCE}")

if not image_path.exists():
    print(f"Image not found: {image_path}")
    sys.exit(1)

if not Path(ENCODINGS_FILE).exists():
    print("Missing encodings.pkl. Run: python train.py")
    sys.exit(1)

OUTPUT_DIR.mkdir(exist_ok=True)

with open(ENCODINGS_FILE, "rb") as file:
    data = pickle.load(file)

known_encodings = data["encodings"]
known_names = data["names"]

rgb_image = face_recognition.load_image_file(image_path)

face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

print(f"Found {len(face_locations)} face(s).")

for face_encoding, face_location in zip(face_encodings, face_locations):
    name = "Unknown"

    distances = face_recognition.face_distance(known_encodings, face_encoding)

    if len(distances) > 0:
        best_match_index = distances.argmin()
        best_distance = distances[best_match_index]

        if best_distance <= TOLERANCE:
            name = known_names[best_match_index]

        print(f"{name} | distance: {best_distance:.2f}")

    top, right, bottom, left = face_location

    cv2.rectangle(bgr_image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(bgr_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    cv2.putText(
        bgr_image,
        name,
        (left + 6, bottom - 6),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

output_path = OUTPUT_DIR / f"recognized_{image_path.name}"
cv2.imwrite(str(output_path), bgr_image)

print(f"Saved result to: {output_path}")