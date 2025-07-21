# mamba_tracker.py
import cv2

class MambaTracker:
    def __init__(self):
        print("[INFO] MambaTracker initialized on CPU.")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def track(self, frames):
        tracked_boxes = []
        for idx, frame in enumerate(frames):
            boxes, _ = self.hog.detectMultiScale(frame,
                                                 winStride=(8, 8),
                                                 padding=(16, 16),
                                                 scale=1.05)
            # Convert to list of tuples: (x, y, w, h)
            boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]
            tracked_boxes.append(boxes)
        return tracked_boxes
