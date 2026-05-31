import subprocess

print("Training face recognition AI...")
subprocess.run(["python3", "train.py"])

print("\nRecognizing faces...")
subprocess.run(["python3", "recognize.py", "unknown/unknown_test.jpg"])