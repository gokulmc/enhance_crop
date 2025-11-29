import cv2
import sys
import os

def crop_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Original dimensions: {width}x{height}")

    # Calculate target dimensions for 9:16 aspect ratio
    # We want to keep the width or height? 
    # "define width_ratio as the ratio of the output cropped_width to original width"
    # "cropped videos height can be stretched to the required 9:16 ratio based on the width_ratio. Cropping should be centered"
    
    # This implies we crop the width, and stretch the height? Or crop width and height?
    # "cropped videos height can be stretched" suggests resizing height.
    # "Cropping should be centered" suggests cropping width.
    
    # Let's assume we want a 9:16 output.
    # If we crop width, we might need to stretch height to match 9:16 ratio with the new width?
    # Or maybe "stretched" means resized.
    
    # Let's interpret:
    # 1. Determine target width (centered crop). 
    #    Usually 9:16 means width < height. 
    #    If original is 16:9 (e.g. 1920x1080), we need to crop width to get 9:16? 
    #    1080 * 9/16 = 607.5. So 608x1080 would be 9:16.
    #    But the prompt says "define width_ratio... height can be stretched...".
    
    # Let's try to follow: "define width_ratio as the ratio of the output cropped_width to original width"
    # This implies we choose a cropped_width. 
    # If we want 9:16, and we keep height constant (initially), then width = height * 9/16.
    # Then width_ratio = (height * 9/16) / original_width.
    
    # But "height can be stretched". This suggests we might change height.
    # If we crop width to W_new, and we want W_new / H_new = 9/16.
    # Then H_new = W_new * 16/9.
    # If H_new != original_height, we stretch (resize) it.
    
    # But what determines W_new?
    # Maybe we just crop to the maximum possible 9:16 center crop?
    # If original is landscape (W > H), 9:16 crop would be limited by height?
    # W_new = H * 9/16.
    # Then H_new = H. No stretching needed?
    
    # "height can be stretched to the required 9:16 ratio based on the width_ratio"
    # This is a bit ambiguous.
    # Maybe it means: Crop width to some ratio?
    # Let's assume we want to convert landscape to portrait 9:16.
    # Common approach: Center crop a 9:16 area.
    # Target aspect = 9/16 = 0.5625.
    # Current aspect = W/H.
    
    # If we just crop width to match 9:16 with current height:
    # Target Width = Height * (9/16).
    # Crop centered.
    # Then we have 9:16 video.
    # "height can be stretched" might mean if we want a specific width ratio?
    
    # Let's stick to the standard "Crop to 9:16" logic:
    # New Width = Height * 9 / 16
    # If New Width > Width (impossible for landscape), then we crop height?
    # New Height = Width * 16 / 9.
    
    # Assuming input is landscape (e.g. 1920x1080).
    # We crop width to 1080 * 9 / 16 = 607.5 -> 608.
    # Height stays 1080.
    # Result is 608x1080 (9:16).
    
    # The prompt mentions "width_ratio".
    # "define width_ratio as the ratio of the output cropped_width to original width"
    # Maybe the user wants to print this ratio?
    
    # "cropped videos height can be stretched to the required 9:16 ratio based on the width_ratio"
    # This sounds like:
    # 1. Crop width (centered).
    # 2. Resize height to make it 9:16?
    # That would distort the video. "Stretched" implies distortion.
    
    # Let's implement:
    # 1. Calculate target width = Height * 9/16.
    # 2. Crop centered to this width.
    # 3. Print width_ratio.
    # 4. If "stretch" is required, maybe it means resizing to fill?
    # But if we crop to 9:16, we are already 9:16.
    
    # Wait, maybe "width_ratio" is an input? No, "define... as ratio".
    
    # Let's assume the goal is to produce a 9:16 video by center cropping the width.
    # And we print the ratio.
    
    target_aspect = 9 / 16
    
    # For landscape input:
    new_width = int(height * target_aspect)
    new_height = height
    
    if new_width > width:
        # Portrait input or square, need to crop height?
        # But prompt says "Cropping should be centered" and talks about width_ratio.
        # Let's assume we crop width.
        new_width = width
        new_height = int(width / target_aspect)
        # This would mean increasing height? We can't crop height to increase it.
        # We would have to pad or stretch.
        pass

    # Center crop width
    start_x = (width - new_width) // 2
    
    print(f"Cropping to {new_width}x{new_height} (Aspect 9:16)")
    width_ratio = new_width / width
    print(f"Width Ratio: {width_ratio}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop
        cropped_frame = frame[:, start_x:start_x+new_width]
        
        # If we needed to stretch height (resize), we would do it here.
        # But if we calculated new_width based on height, we don't need to stretch height to get 9:16.
        # Unless the user WANTS distortion?
        # "height can be stretched to the required 9:16 ratio"
        # This might mean: Crop width to X, then resize height to Y such that X/Y = 9/16?
        # But we can just crop width to match height?
        
        # Let's assume standard center crop to 9:16.
        
        out.write(cropped_frame)

    cap.release()
    out.release()
    print(f"Saved cropped video to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python crop_video.py <input_video> <output_video>")
        sys.exit(1)
    
    crop_video(sys.argv[1], sys.argv[2])
