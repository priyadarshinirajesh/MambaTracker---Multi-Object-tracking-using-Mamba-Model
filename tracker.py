import cv2
import os
import numpy as np
from utils import draw_boxes, cosine_similarity
from mamba_feature_extractor import extract_features  # Your embedding function

class Track:
    def __init__(self, track_id, bbox, feature):
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature

def load_detections(det_path):
    detections = {}
    with open(det_path, 'r') as f:
        for line in f:
            frame_id, _, x, y, w, h, _, _, _, _ = map(float, line.strip().split(','))
            frame_id = int(frame_id)
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append([x, y, w, h])
    return detections

def crop_bbox(image, bbox):
    x, y, w, h = map(int, bbox)
    return image[y:y+h, x:x+w]

def tracker_pipeline(image_folder, det_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    detections = load_detections(det_file)
    track_id_counter = 0
    active_tracks = []

    image_files = sorted(os.listdir(image_folder))
    for frame_idx, image_name in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        if frame_idx not in detections:
            continue

        frame_dets = detections[frame_idx]
        crops = [crop_bbox(image, det) for det in frame_dets]
        features = extract_features(crops)

        new_tracks = []
        for i, (bbox, feat) in enumerate(zip(frame_dets, features)):
            matched = False
            for track in active_tracks:
                sim = cosine_similarity(track.feature, feat)
                if sim > 0.7:
                    track.bbox = bbox
                    track.feature = feat
                    new_tracks.append(track)
                    matched = True
                    break
            if not matched:
                new_tracks.append(Track(track_id_counter, bbox, feat))
                track_id_counter += 1

        active_tracks = new_tracks

        # ✅ Draw and Save the visualized frame
        draw_boxes(image, active_tracks)
        out_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, image)

    print("✅ Tracking done and frames saved to", output_folder)

if __name__ == "__main__":
    image_folder = "data/test/dancetrack0003/img1"
    det_file = "data/test/dancetrack0003/det/det.txt"
    output_folder = "data/test/dancetrack0003/tracks"

    tracker_pipeline(image_folder, det_file, output_folder)
