import os
import cv2

def load_frames_from_sequence(sequence_dir):
    """Load sorted image frames from a sequence directory"""
    frame_files = sorted([
        os.path.join(sequence_dir, f)
        for f in os.listdir(sequence_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    frames = [cv2.imread(f) for f in frame_files]
    return frames
