import cv2
import sys
import os

file_path = "input/1.mp4"
print(f"Testing file: {os.path.abspath(file_path)}")

if not os.path.exists(file_path):
    print("File does not exist!")
    sys.exit(1)

cap = cv2.VideoCapture(file_path)
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Success: Video opened.")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    cap.release()
