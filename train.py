import face_recognition
import pickle
from pathlib import Path

KNOWN_DIR = Path("known")
OUTPUT_FILE = "encodings.pkl"

known_encodings = []
known_names = []

image_types = ["*.jpg", "*.jpeg", "*.png"]

for image_type in image_types:
    for image_path in KNOWN_DIR.glob(image_type):
        print(f"Learning face from: {image_path.name}")

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"No face found in {image_path.name}, skipping.")
            continue

        known_encodings.append(encodings[0])
        known_names.append(image_path.stem)

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(OUTPUT_FILE, "wb") as file:
    pickle.dump(data, file)

print(f"Training complete. Learned {len(known_names)} face(s).")