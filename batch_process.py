import os
import time
import csv
import subprocess
from datetime import datetime
import sys

def batch_process(input_dir, output_dir, csv_file):
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CSV
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Filename', 'Start Time', 'End Time', 'Duration (s)', 'Status'])
            
    # Get list of video files
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    video_files.sort()
    
    total_videos = len(video_files)
    print(f"Found {total_videos} videos to process.")
    
    for i, filename in enumerate(video_files):
        print(f"\n[{i+1}/{total_videos}] Processing {filename}...")
        
        input_path = os.path.join(input_dir, filename)
        name, _ = os.path.splitext(filename)
        enhanced_temp_path = os.path.join(output_dir, f"{name}_enhanced_temp.mp4")
        final_cropped_path = os.path.join(output_dir, "cropped", f"{name}_cropped.mp4")
        
        start_time = time.time()
        start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if already processed
        if os.path.exists(final_cropped_path):
            print(f"  > Skipping {filename} (already exists at {final_cropped_path})")
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, start_str, start_str, "0", "Skipped"])
            continue
        
        try:
            # 1. Enhance
            print("  > Running Enhancement...")
            # Note: We use the venv python to run the backend
            cmd_enhance = [
                "venv/bin/python", "backend/rve-backend.py",
                "-i", input_path,
                "-o", enhanced_temp_path,
                "--upscale_model", "models/2x_OpenProteus_Compact_i2_70K.pth",
                "--backend", "pytorch",
                "--overwrite"
            ]
            subprocess.run(cmd_enhance, check=True)
            
            # 2. Process (Audio Merge & Crop)
            print("  > Running Post-processing (Audio & Crop)...")
            cmd_process = [
                "venv/bin/python", "process_video.py",
                input_path,
                enhanced_temp_path,
                output_dir
            ]
            subprocess.run(cmd_process, check=True)
            
            # Clean up temp file
            if os.path.exists(enhanced_temp_path):
                os.remove(enhanced_temp_path)
                
            end_time = time.time()
            end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration = end_time - start_time
            
            print(f"  > Completed in {duration:.2f} seconds.")
            
            # Log to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, start_str, end_str, f"{duration:.2f}", "Success"])
                
        except subprocess.CalledProcessError as e:
            print(f"  > Error processing {filename}: {e}")
            # Log error
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, start_str, "ERROR", "ERROR", f"Error: {str(e)}"])
        except Exception as e:
            print(f"  > Unexpected error processing {filename}: {e}")
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, start_str, "ERROR", "ERROR", f"Unexpected Error: {str(e)}"])
                
if __name__ == "__main__":
    input_dir = "../input"
    output_dir = "../output"
    csv_file = "processing_times.csv"
    
    # Adjust paths if running from inside REAL-Video-Enhancer
    if os.path.basename(os.getcwd()) == "REAL-Video-Enhancer":
        input_dir = "../input"
        output_dir = "../output"
    else:
        # Assume running from parent dir, but script is in REAL-Video-Enhancer
        # This case might need adjustment depending on where user runs it
        pass

    batch_process(input_dir, output_dir, csv_file)
