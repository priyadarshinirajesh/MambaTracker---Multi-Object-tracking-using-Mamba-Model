import os
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import read_seqinfo
import gc

def visualize_sequence(seq_path, results_txt, output_dir="results/visualizations"):
    print(f"Visualizing sequence from {seq_path}")
    os.makedirs(output_dir, exist_ok=True)
    seq = read_seqinfo(seq_path)
    print(f"Sequence info: {seq}")
    img_dir = os.path.join(seq_path, seq["imDir"])
    fps = seq["frameRate"]
    size = (seq["imWidth"], seq["imHeight"])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(os.path.join(output_dir, f"{seq['seqName']}.mp4"), fourcc, fps, size)
    print(f"Video writer initialized for {seq['seqName']}.mp4")

    results = pd.read_csv(results_txt)
    print(f"Loaded results from {results_txt} with shape: {results.shape}")
    for f in range(1, seq["seqLength"] + 1):
        print(f"Processing frame {f}")
        img = cv2.imread(os.path.join(img_dir, f"{f:06d}{seq['imExt']}"))
        if img is None:
            print(f"Warning: Image not found for frame {f}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fr = results[results["frame"] == f]
        print(f"Found {len(fr)} detections for frame {f}")
        for _, row in fr.iterrows():
            x, y, w, h, iid = row[["x", "y", "w", "h", "id"]].astype(int)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"ID:{iid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        plt.imsave(os.path.join(output_dir, f"{f:06d}.png"), img)
        del img, fr
        gc.collect()
    video.release()
    print(f"Video {seq['seqName']}.mp4 saved")

def plot_losses():
    print("Plotting losses")
    d = torch.load("results/losses.pth")
    print(f"Loaded losses data: train_losses length {len(d['train_losses'])}, val_losses length {len(d['val_losses'])}")
    plt.figure(figsize=(8, 5))
    plt.plot(d["train_losses"], label="train")
    plt.plot(d["val_losses"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/loss_plot.png")
    plt.close()
    print("Loss plot saved to results/loss_plot.png")

if __name__ == "__main__":
    seq = "data/val/dancetrack0001"
    visualize_sequence(seq, "results/val_results.txt")
    plot_losses()