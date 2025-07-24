import os
import cv2
import torch
import yaml
from tqdm import tqdm
from types import SimpleNamespace
from mambapy.mamba import Mamba

# ---------- Step 1: Load YAML Config ----------
CONFIG_PATH = "mamba_config.yaml"  # Make sure this path is correct

with open(CONFIG_PATH, "r") as f:
    config_dict = yaml.safe_load(f)

# Convert config dict to object so we can use dot notation like config.n_layers
config = SimpleNamespace(**config_dict)

# ---------- Step 2: Load Mamba Model ----------
device = torch.device("cpu")
model = Mamba(config).to(device)
model.eval()

# ---------- Step 3: Load a video or dataset ----------
VIDEO_PATH = "data\\test1\\dancetrack0003\\img1"  # Change this to your video path
cap = cv2.VideoCapture(VIDEO_PATH)

# Optional: Define your tracker state if Mamba provides one (depends on implementation)
tracker_state = None

# ---------- Step 4: Object Tracking Loop ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame if needed (e.g., resize or normalize)
    input_frame = cv2.resize(frame, (config.input_dim, config.input_dim))
    input_tensor = torch.from_numpy(input_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Pass through model
    with torch.no_grad():
        output = model(input_tensor)

    # TODO: Parse the output to draw bounding boxes, labels, etc.
    # Placeholder visualization
    cv2.putText(frame, "Tracking...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Mamba Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
