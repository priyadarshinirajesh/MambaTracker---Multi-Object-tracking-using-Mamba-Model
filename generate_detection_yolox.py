import cv2
import os
from yolox.exp import get_exp
from yolox.utils import postprocess
import torch

def load_model():
    exp = get_exp(exp_file=None, name="yolox-s")  # Or yolox-nano
    model = exp.get_model()
    ckpt = torch.load("weights/yolox_s.pth", map_location="cpu")  # Path to model
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def detect_images(model, image_dir, output_txt):
    with open(output_txt, 'w') as f:
        for idx, img_file in enumerate(sorted(os.listdir(image_dir))):
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            # Resize, normalize, and convert to tensor here (based on YOLOX preproc)
            # Run model
            # Get boxes, scores, class_ids
            # Write frame_idx, x1, y1, w, h, score, class to output_txt
