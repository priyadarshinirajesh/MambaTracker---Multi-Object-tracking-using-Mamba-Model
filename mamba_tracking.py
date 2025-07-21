import os
import cv2
from utils.preprocess import load_frames_from_sequence
from mamba_tracker import MambaTracker

def run_tracking_on_sequence(sequence_path):
    print(f"[INFO] Starting tracking on sequence: {sequence_path}")

    frames = load_frames_from_sequence(sequence_path)
    print(f"[DEBUG] Loaded {len(frames)} frames from {sequence_path}")

    if not frames:
        print("[ERROR] No frames loaded. Check your image directory.")
        return

    tracker = MambaTracker()
    print("[DEBUG] Tracker initialized.")

    tracked_boxes = tracker.track(frames)
    print("[DEBUG] Tracking completed.")

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 800, 600)

    for idx, frame in enumerate(frames):
        print(f"[DEBUG] Processing frame {idx + 1}/{len(frames)}")
        boxes = tracked_boxes[idx]
        print(f"[DEBUG] Frame {idx + 1}: Detected {len(boxes)} person(s)")

        for i, box in enumerate(boxes):
            x, y, w, h = box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID {i}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show each frame
        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(50)  # Adjust delay as needed
        if key == 27:  # ESC key
            print("[INFO] ESC pressed. Exiting early.")
            break

    print("[INFO] Tracking visualization complete.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_dir = os.path.join("data", "test1", "dancetrack0040", "img1")
    print(f"[INFO] Using test directory: {test_dir}")
    run_tracking_on_sequence(test_dir)
