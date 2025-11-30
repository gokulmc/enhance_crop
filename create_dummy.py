import cv2
import numpy as np

width, height = 1920, 1080
fps = 30
duration = 2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dummy.mp4', fourcc, fps, (width, height))

for _ in range(fps * duration):
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    out.write(frame)

out.release()
print("Created dummy.mp4")
