import os
import time
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # Run in headless mode if needed, but visible is better for debugging
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def login(driver, username, password):
    print("Navigating to login page...")
    driver.get("http://lpq.qtvai.in")
    
    try:
        # Wait for username field
        print("Waiting for login form...")
        username_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )
        password_field = driver.find_element(By.ID, "password")
        login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Login') or contains(text(), 'Sign in')]")
        
        print(f"Logging in as {username}...")
        username_field.clear()
        username_field.send_keys(username)
        password_field.clear()
        password_field.send_keys(password)
        login_button.click()
        
        # Wait for redirect or dashboard element
        print("Login submitted. Waiting for redirection...")
        try:
            WebDriverWait(driver, 15).until(
                lambda d: "auth" not in d.current_url
            )
        except:
            print("Timed out waiting for URL change.")
            
        print(f"Current URL: {driver.current_url}")
        
        # Check if login failed
        if "auth" in driver.current_url:
            print("Warning: Still on auth page. Login might have failed.")
            driver.save_screenshot("login_failed.png")
            return False
            
        print("Login successful.")
        return True
        
    except Exception as e:
        print(f"Login failed: {e}")
        driver.save_screenshot("login_exception.png")
        return False

def get_file_pairs(cropped_dir, thumbnail_dir):
    pairs = []
    if not os.path.exists(cropped_dir):
        print(f"Error: Cropped directory not found: {cropped_dir}")
        return pairs
        
    cropped_files = [f for f in os.listdir(cropped_dir) if f.endswith('.mp4')]
    cropped_files.sort()
    
    print(f"Found {len(cropped_files)} cropped videos.")
    
    for video_file in cropped_files:
        # Extract prefix: "1_cropped.mp4" -> "1"
        prefix = video_file.split('_')[0]
        
        # Look for thumbnail
        thumb_candidates = [
            f"{prefix}.jpg",
            f"{prefix}_cropped.jpg",
            f"{prefix}.png"
        ]
        
        thumb_path = None
        for cand in thumb_candidates:
            path = os.path.join(thumbnail_dir, cand)
            if os.path.exists(path):
                thumb_path = path
                break
        
        if thumb_path:
            pairs.append({
                'video': os.path.join(cropped_dir, video_file),
                'thumbnail': thumb_path,
                'name': video_file
            })
        else:
            print(f"Warning: No thumbnail found for {video_file} (Prefix: {prefix})")
            
    return pairs

def upload_files(driver, pairs, dry_run=False):
    print(f"Starting upload for {len(pairs)} pairs...")
    
    for i, pair in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] Processing {pair['name']}...")
        
        if dry_run:
            print(f"  [Dry Run] Would upload:\n    Video: {pair['video']}\n    Thumb: {pair['thumbnail']}")
            continue
            
        try:
            # Ensure we are on the upload page (or dashboard where upload is)
            # Based on inspection, it seems to be the main page after login
            
            # Find inputs
            video_input = driver.find_element(By.ID, "video-upload")
            thumb_input = driver.find_element(By.ID, "thumbnail-upload")
            
            print("  Sending file paths...")
            video_input.send_keys(os.path.abspath(pair['video']))
            thumb_input.send_keys(os.path.abspath(pair['thumbnail']))
            
            # Debug: Screenshot after selecting files
            time.sleep(2)
            driver.save_screenshot(f"debug_{pair['name']}_selected.png")
            
            # Try to find a button that looks like "Upload" or "Submit"
            try:
                # Wait for button to be clickable
                print("  Waiting for upload button to be clickable...")
                upload_btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Upload') or contains(text(), 'Submit') or contains(text(), 'Save')]"))
                )
                
                print(f"  Button found: '{upload_btn.text}'. Clicking...")
                upload_btn.click()
                
                # Wait for success or some change
                time.sleep(5)
                driver.save_screenshot(f"debug_{pair['name']}_clicked.png")
                
            except Exception as e:
                print(f"  Standard click failed or button not found: {e}")
                print("  Attempting JS click...")
                try:
                    upload_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Upload') or contains(text(), 'Submit')]")
                    driver.execute_script("arguments[0].click();", upload_btn)
                    print("  JS Click sent.")
                    time.sleep(5)
                    driver.save_screenshot(f"debug_{pair['name']}_js_clicked.png")
                except Exception as js_e:
                    print(f"  JS click also failed: {js_e}")
            
            # Wait for success (URL change)
            print("  Waiting for upload to complete (URL change)...")
            try:
                WebDriverWait(driver, 60).until(
                    lambda d: d.current_url != "https://lpq.qtvai.in/" and d.current_url != "http://lpq.qtvai.in/"
                )
                print("  Upload successful (redirected).")
            except:
                print("  Warning: URL did not change. Upload might have failed or is taking too long.")
                driver.save_screenshot(f"debug_{pair['name']}_timeout.png")
            
            # Navigate back to main page for next upload
            print("  Returning to main page...")
            driver.get("https://lpq.qtvai.in/")
            
            # Wait for upload inputs to be ready again
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "video-upload"))
            )
            
        except Exception as e:
            print(f"  Error uploading {pair['name']}: {e}")
            # Take screenshot on error
            driver.save_screenshot(f"error_{pair['name']}.png")
            # Try to recover by going back to main page
            try:
                driver.get("https://lpq.qtvai.in/")
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description="Upload videos and thumbnails to QTV.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate upload without actions")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust paths based on where script is running
    if os.path.basename(base_dir) == "REAL-Video-Enhancer":
        cropped_dir = os.path.join(base_dir, "../output/cropped")
        thumbnail_dir = os.path.join(base_dir, "../output/thumbnail")
    else:
        # Fallback or assume standard structure
        cropped_dir = os.path.join(base_dir, "output/cropped")
        thumbnail_dir = os.path.join(base_dir, "output/thumbnail")

    print(f"Video Dir: {cropped_dir}")
    print(f"Thumb Dir: {thumbnail_dir}")

    pairs = get_file_pairs(cropped_dir, thumbnail_dir)
    
    if args.limit > 0:
        pairs = pairs[:args.limit]
        
    if not pairs:
        print("No files to upload.")
        return

    driver = setup_driver()
    try:
        if login(driver, "qtv3", "qsignage@QTV3"):
            upload_files(driver, pairs, dry_run=args.dry_run)
        else:
            print("Aborting due to login failure.")
    finally:
        print("Closing driver...")
        driver.quit()

if __name__ == "__main__":
    main()
