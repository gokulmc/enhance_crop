import cv2
import os
import sys
import numpy as np
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

def get_scenes(video_path):
    """
    Detects scenes in a video using PySceneDetect.
    Returns a list of (start_frame, end_frame) tuples.
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()

    scenes = []
    for scene in scene_list:
        start, end = scene
        scenes.append((start.get_frames(), end.get_frames()))
    
    return scenes

def calculate_colorfulness(image):
    """
    Calculates colorfulness using Hasler and Suesstrunk's metric.
    """
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    
    return stdRoot + (0.3 * meanRoot)

def calculate_entropy(image):
    """
    Calculates the Shannon entropy of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    logs = np.log2(hist + 1e-7)
    entropy = -1 * (hist * logs).sum()
    return entropy

def calculate_frame_score(frame):
    """
    Calculates a quality score for a frame based on sharpness, contrast, saturation, colorfulness, and entropy.
    """
    if frame is None:
        return 0

    # 1. Sharpness (Variance of Laplacian)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Contrast (Standard Deviation of Gray)
    contrast = gray.std()

    # 3. Saturation (Mean of S channel in HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    
    # 4. Colorfulness
    colorfulness = calculate_colorfulness(frame)
    
    # 5. Entropy
    entropy = calculate_entropy(frame)

    # Normalize roughly to combine (weights can be tuned)
    # Sharpness usually 0-1000+, Contrast 0-100, Saturation 0-255
    # Colorfulness 0-100+, Entropy 0-8
    
    # Weighting to favor "rich" images over text
    # Entropy and Colorfulness get higher weights
    
    score = (sharpness / 100.0) + (contrast / 10.0) + (saturation / 20.0) + (colorfulness * 2.0) + (entropy * 10.0)
    return score

def extract_best_thumbnail(video_path, output_dir):
    """
    Extracts the best thumbnail from the video and saves it to output_dir.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.jpg")

    # Check if already exists
    if os.path.exists(output_path):
        print(f"Skipping {filename} (Thumbnail already exists at {output_path})")
        return

    print(f"Processing {filename}...")

    # 1. Detect Scenes
    print("  Detecting scenes...")
    try:
        scenes = get_scenes(video_path)
    except Exception as e:
        print(f"  Scene detection failed: {e}. Fallback to uniform sampling.")
        scenes = []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    candidate_frames = []
    
    if scenes:
        print(f"  Found {len(scenes)} scenes. Sampling middle frames.")
        for start, end in scenes:
            mid_frame = (start + end) // 2
            candidate_frames.append(mid_frame)
    else:
        print("  No scenes detected or single scene. Sampling at 20%, 50%, 80%.")
        candidate_frames = [
            int(total_frames * 0.2),
            int(total_frames * 0.5),
            int(total_frames * 0.8)
        ]

    best_score = -1
    best_frame = None
    best_frame_idx = -1

    print(f"  Analyzing {len(candidate_frames)} candidate frames...")
    
    for frame_idx in candidate_frames:
        if frame_idx >= total_frames:
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        score = calculate_frame_score(frame)
        # print(f"    Frame {frame_idx}: Score {score:.2f}")
        
        if score > best_score:
            best_score = score
            best_frame = frame
            best_frame_idx = frame_idx

    cap.release()

    if best_frame is not None:
        cv2.imwrite(output_path, best_frame)
        print(f"  Saved best thumbnail (Frame {best_frame_idx}, Score {best_score:.2f}) to {output_path}")
    else:
        print("  Could not extract any frames.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to ../output/cropped if no argument provided
        input_path = "../output/cropped"
        print(f"No input provided. Defaulting to {input_path}")
    else:
        input_path = sys.argv[1]

    output_base = "../output/thumbnail"

    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)

    if os.path.isfile(input_path):
        extract_best_thumbnail(input_path, output_base)
    elif os.path.isdir(input_path):
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        files = [f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in video_extensions]
        files.sort()
        print(f"Found {len(files)} videos in {input_path}")
        for f in files:
            extract_best_thumbnail(os.path.join(input_path, f), output_base)
    else:
        print(f"Invalid input: {input_path}")
