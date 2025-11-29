import subprocess
import sys
import os
import json

def get_video_info(file_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return int(info['streams'][0]['width']), int(info['streams'][0]['height'])

def process_video(original_path, enhanced_temp_path, output_dir):
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)
    
    enhanced_dir = os.path.join(output_dir, "enhanced")
    cropped_dir = os.path.join(output_dir, "cropped")
    
    os.makedirs(enhanced_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)
    
    final_enhanced_path = os.path.join(enhanced_dir, f"{name}_enhanced.mp4")
    final_cropped_path = os.path.join(cropped_dir, f"{name}_cropped.mp4")
    
    # 1. Get original dimensions
    orig_w, orig_h = get_video_info(original_path)
    print(f"Original Dimensions: {orig_w}x{orig_h}")
    
    target_w = orig_w * 2
    target_h = orig_h * 2
    print(f"Target Enhanced Dimensions: {target_w}x{target_h}")
    
    # 2. Resize and Merge Audio
    # -i enhanced_temp (video)
    # -i original (audio)
    # scale to target_w:target_h
    # map video from 0, audio from 1
    print(f"Creating enhanced video with audio at {final_enhanced_path}...")
    cmd_enhance = [
        "ffmpeg", "-y",
        "-i", enhanced_temp_path,
        "-i", original_path,
        "-map", "0:v", "-map", "1:a?", # Use audio from original if exists
        "-vf", f"scale={target_w}:{target_h}",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        final_enhanced_path
    ]
    subprocess.run(cmd_enhance, check=True)
    
    # 3. Crop and Stretch to 9:16
    # New Requirement: Width Ratio = 90%
    # Crop width = 90% of enhanced width
    # Stretch height to achieve 9:16 aspect ratio based on the new width
    
    crop_w = int(target_w * 0.9)
    # Ensure even dimensions
    if crop_w % 2 != 0:
        crop_w -= 1
        
    # Calculate final height for 9:16 aspect ratio
    final_h = int(crop_w * 16 / 9)
    if final_h % 2 != 0:
        final_h -= 1
        
    print(f"Cropping to width {crop_w} (90% of {target_w}) and stretching height to {final_h} (9:16)...")
    
    # Center crop first, then scale
    # crop=w:h:x:y
    # We crop width to crop_w, keep height as target_h (for now)
    # Then scale to crop_w:final_h
    
    cmd_crop = [
        "ffmpeg", "-y",
        "-i", final_enhanced_path,
        "-vf", f"crop={crop_w}:{target_h}:(in_w-{crop_w})/2:(in_h-{target_h})/2,scale={crop_w}:{final_h}",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-c:a", "copy",
        final_cropped_path
    ]
    subprocess.run(cmd_crop, check=True)
    
    print("Processing complete!")
    print(f"Enhanced: {final_enhanced_path}")
    print(f"Cropped: {final_cropped_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python process_video.py <original_video> <enhanced_temp_video> <output_dir>")
        sys.exit(1)
        
    process_video(sys.argv[1], sys.argv[2], sys.argv[3])
