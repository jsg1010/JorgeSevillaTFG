import cv2
import os

def extract_frames(video_path, output_folder, interval=10):
    """
    Extracts frames from a video every `interval` seconds and saves them as images.

    :param video_path: Path to the input video file.
    :param output_folder: Folder where extracted frames will be saved.
    :param interval: Time interval (in seconds) between frames.
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # Duration in seconds
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    frame_interval = fps * interval  # Frame interval to extract images
    frame_count = 0
    image_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_folder, f"frame_{image_count:04d}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved: {image_path}")
            image_count += 1
        
        frame_count += 1
    
    cap.release()
    print("Frame extraction complete.")

input_video = "C:/Users/TFG/Desktop/train/video.mp4"
output_folder = "C:/Users/TFG/Desktop/train/Video_Frames"
extract_frames(input_video, output_folder, interval=10)
