import cv2
import os

def save_frames_on_space(input_path, output_path):
    # Create output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break
        
        frame_count += 1
        cv2.imshow("Video Frame", frame)

        key = cv2.waitKey(0)  # Wait for a key press

        if key == 32:  # Spacebar key
            frame_filename = os.path.join(output_path, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
            saved_count += 1
        elif key == 27:  # ESC key to exit
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_path = "records/closed_hand_mask.mp4"  # Change this to your video file
    output_path = 'data/HandGestures/closed/train'
    save_frames_on_space(input_path, output_path)
