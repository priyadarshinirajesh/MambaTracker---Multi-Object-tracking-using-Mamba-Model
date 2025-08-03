import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import DancetrackDataset
from src.model import MambaTracker
import os
import gc

def compute_loss(pred_bboxes, pred_ids, gt_bboxes, gt_ids, device):
    print(f"compute_loss - pred_bboxes shape: {pred_bboxes.shape}, pred_ids shape: {pred_ids.shape}, gt_bboxes shape: {gt_bboxes.shape}, gt_ids shape: {gt_ids.shape}")
    batch_size, _, height, width = pred_bboxes.shape
    pred_bboxes = pred_bboxes.permute(0, 2, 3, 1).reshape(batch_size, height * width, 5)  # [batch_size, height*width, 5]
    print(f"Reshaped pred_bboxes: {pred_bboxes.shape}")
    conf_scores = F.sigmoid(pred_bboxes[:, :, 4])
    print(f"Confidence scores shape: {conf_scores.shape}, min: {conf_scores.min()}, max: {conf_scores.max()}")
    mask = conf_scores > 0.5
    print(f"Mask shape: {mask.shape}, True count: {mask.sum()}")

    l1_loss = torch.tensor(0.0, device=device, requires_grad=True)
    conf_loss = torch.tensor(0.0, device=device, requires_grad=True)
    id_loss = torch.tensor(0.0, device=device, requires_grad=True)

    if gt_bboxes.numel() > 0:
        pred_bboxes_flat = pred_bboxes[mask]
        print(f"Filtered pred_bboxes_flat shape: {pred_bboxes_flat.shape}")
        if pred_bboxes_flat.numel() > 0:
            num_gt = gt_bboxes.shape[1]
            num_pred = pred_bboxes_flat.shape[0]
            print(f"num_gt: {num_gt}, num_pred: {num_pred}")
            num_match = min(num_pred, num_gt)
            if num_match > 0:
                pred_bboxes_matched = pred_bboxes_flat[:num_match, :4]
                gt_bboxes_matched = gt_bboxes[:, :num_match, :].reshape(-1, 4).to(device)
                print(f"Matched shapes - pred: {pred_bboxes_matched.shape}, gt: {gt_bboxes_matched.shape}")
                l1_loss = F.l1_loss(pred_bboxes_matched, gt_bboxes_matched)
                conf_loss = F.binary_cross_entropy_with_logits(pred_bboxes_flat[:num_match, 4], torch.ones(num_match, device=device))
            else:
                pred_mean = pred_bboxes.mean(dim=1, keepdim=True)  # [batch_size, 1, 5]
                gt_mean = gt_bboxes.mean(dim=1, keepdim=True).to(device)  # [batch_size, 1, 4]
                print(f"Fallback shapes - pred_mean: {pred_mean.shape}, gt_mean: {gt_mean.shape}")
                l1_loss = F.l1_loss(pred_mean[:, :, :4], gt_mean)
                conf_loss = F.binary_cross_entropy_with_logits(pred_mean[:, :, 4].mean(), torch.tensor(1.0, device=device))
        else:
            pred_mean = pred_bboxes.mean(dim=1, keepdim=True)  # [batch_size, 1, 5]
            gt_mean = gt_bboxes.mean(dim=1, keepdim=True).to(device)  # [batch_size, 1, 4]
            print(f"No boxes fallback - pred_mean: {pred_mean.shape}, gt_mean: {gt_mean.shape}")
            l1_loss = F.l1_loss(pred_mean[:, :, :4], gt_mean)
            conf_loss = F.binary_cross_entropy_with_logits(pred_bboxes[:, :, 4].mean(), torch.tensor(1.0, device=device))

    if gt_ids.numel() > 0 and pred_ids.numel() > 0:
        id_loss = F.mse_loss(pred_ids, torch.zeros_like(pred_ids).to(device))
        print(f"ID loss computed with shapes: {pred_ids.shape}, {torch.zeros_like(pred_ids).shape}")
    else:
        id_loss = torch.tensor(0.0, device=device, requires_grad=True)
        print("No ID loss computed due to empty tensors")

    total_loss = l1_loss + conf_loss + id_loss
    print(f"Total loss: {total_loss.item()}, requires_grad: {total_loss.requires_grad}")
    return total_loss

def train(data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = MambaTracker().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print(f"Optimizer setup with learning rate: {1e-4}")
    num_epochs = 10
    print(f"Training for {num_epochs} epochs")

    train_dataset = DancetrackDataset(data_root, split="train", img_size=(320, 320))
    val_dataset = DancetrackDataset(data_root, split="val", img_size=(320, 320))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}")

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        print(f"Starting epoch {epoch + 1}")
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            print(f"Processing batch {batch_idx} of epoch {epoch + 1}")
            images = batch["image"].to(device)
            gt_bboxes = batch["boxes"].to(device)
            gt_ids = batch["ids"].to(device)
            print(f"Batch shapes - images: {images.shape}, gt_bboxes: {gt_bboxes.shape}, gt_ids: {gt_ids.shape}")
            optimizer.zero_grad()
            pred_bboxes, pred_ids = model(images)
            loss = compute_loss(pred_bboxes, pred_ids, gt_bboxes, gt_ids, device)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                print(f"Backward pass and step completed for batch {batch_idx}")
            else:
                print(f"Warning: Loss does not require grad, skipping step for batch {batch_idx}")
            train_loss += loss.item()
            del images, gt_bboxes, gt_ids, pred_bboxes, pred_ids, loss
            torch.cuda.empty_cache()
            gc.collect()

        train_loss /= max(1, len(train_loader))
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1} train loss computed: {train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                print(f"Processing validation batch {batch_idx} of epoch {epoch + 1}")
                images = batch["image"].to(device)
                gt_bboxes = batch["boxes"].to(device)
                gt_ids = batch["ids"].to(device)
                pred_bboxes, pred_ids = model(images)
                loss = compute_loss(pred_bboxes, pred_ids, gt_bboxes, gt_ids, device)
                val_loss += loss.item()
                del images, gt_bboxes, gt_ids, pred_bboxes, pred_ids, loss
                torch.cuda.empty_cache()
                gc.collect()

        val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1} validation loss: {val_loss}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/mamba_tracker.pth")
    torch.save({"train_losses": train_losses, "val_losses": val_losses}, "results/losses.pth")
    print("Model and losses saved to results/")

if __name__ == "__main__":
    train("data")