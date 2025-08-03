import torch
from torch.utils.data import DataLoader
from src.dataset import DancetrackDataset
from src.model import MambaTracker
import pandas as pd
import os
import trackeval
import torch.nn.functional as F
import gc

def test(data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = MambaTracker().to(device)
    model.load_state_dict(torch.load("results/mamba_tracker.pth"))
    print("Model loaded from results/mamba_tracker.pth")

    test_dataset = DancetrackDataset(data_root, split="test", img_size=(320, 320))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test loader size: {len(test_loader)}")

    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            print(f"Processing test batch {batch_idx}")
            images = batch["image"].to(device)
            pred_bboxes, pred_ids = model(images)
            print(f"Test batch {batch_idx} - images shape: {images.shape}, pred_bboxes shape: {pred_bboxes.shape}, pred_ids shape: {pred_ids.shape}")
            conf = F.sigmoid(pred_bboxes[:, 4]).cpu().numpy()
            mask = conf > 0.5
            print(f"Confidence mask shape: {mask.shape}, True count: {mask.sum()}")
            for i in range(conf.shape[1]):
                if mask[0, i]:
                    results.append([batch_idx + 1, i + 1, *pred_bboxes[0, :4, i].cpu().numpy(), conf[0, i], -1, -1])
            del images, pred_bboxes, pred_ids
            torch.cuda.empty_cache()
            gc.collect()

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results, columns=["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]).to_csv("results/test_results.txt", index=False)
    print(f"Saved {len(results)} test results to results/test_results.txt")

    val_dataset = DancetrackDataset(data_root, split="val", img_size=(320, 320))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Val loader size: {len(val_loader)}")

    val_results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            print(f"Processing val batch {batch_idx}")
            images = batch["image"].to(device)
            pred_bboxes, pred_ids = model(images)
            print(f"Val batch {batch_idx} - images shape: {images.shape}, pred_bboxes shape: {pred_bboxes.shape}, pred_ids shape: {pred_ids.shape}")
            conf = F.sigmoid(pred_bboxes[:, 4]).cpu().numpy()
            mask = conf > 0.5
            print(f"Confidence mask shape: {mask.shape}, True count: {mask.sum()}")
            for i in range(conf.shape[1]):
                if mask[0, i]:
                    val_results.append([batch_idx + 1, i + 1, *pred_bboxes[0, :4, i].cpu().numpy(), conf[0, i], -1, -1])
            del images, pred_bboxes, pred_ids
            torch.cuda.empty_cache()
            gc.collect()

    pd.DataFrame(val_results, columns=["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]).to_csv("results/val_results.txt", index=False)
    print(f"Saved {len(val_results)} val results to results/val_results.txt")

    evaluator = trackeval.Evaluator()
    dataset_config = {
        "GT_FOLDER": os.path.join(data_root, "val"),
        "TRACKERS_FOLDER": "results",
        "TRACKERS_TO_EVAL": ["mamba_tracker"],
        "SEQMAP_FILE": None,
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT20",
        "SPLIT_TO_EVAL": "val",
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": True,
        "OUTPUT_SUB_FOLDER": ""
    }
    dataset = trackeval.datasets.MOTChallenge2DBox(dataset_config)
    metrics = evaluator.evaluate(dataset)
    print("Validation Metrics:", metrics)

    train_dataset_config = {
        "GT_FOLDER": os.path.join(data_root, "train"),
        "TRACKERS_FOLDER": "results",
        "TRACKERS_TO_EVAL": ["mamba_tracker"],
        "SEQMAP_FILE": None,
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT20",
        "SPLIT_TO_EVAL": "train",
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": True,
        "OUTPUT_SUB_FOLDER": ""
    }
    train_dataset = trackeval.datasets.MOTChallenge2DBox(train_dataset_config)
    train_metrics = evaluator.evaluate(train_dataset)
    print("Train Metrics:", train_metrics)

if __name__ == "__main__":
    test("data")