import os
import cv2
import torch
from torch.utils.data import Dataset
from src.utils import read_seqinfo, load_gt

class DancetrackDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None, img_size=(320, 320)):
        self.root = os.path.join(data_root, split)
        self.data = []
        self.transform = transform
        self.img_size = img_size
        print(f"Initializing {split} dataset from {self.root}")
        for seq in sorted(os.listdir(self.root)):
            seq_path = os.path.join(self.root, seq)
            if not os.path.isdir(seq_path):
                print(f"Skipping non-directory: {seq_path}")
                continue
            seq_info = read_seqinfo(seq_path)
            img_dir = os.path.join(seq_path, seq_info["imDir"])
            if split != "test":
                gt = load_gt(os.path.join(seq_path, "gt", "gt.txt"))
                self.data.append({"seq_info": seq_info, "img_dir": img_dir, "gt": gt})
                print(f"Added sequence {seq_info['seqName']} with {seq_info['seqLength']} frames")
            else:
                self.data.append({"seq_info": seq_info, "img_dir": img_dir})
                print(f"Added test sequence {seq_info['seqName']} with {seq_info['seqLength']} frames")

    def __len__(self):
        length = sum(s["seq_info"]["seqLength"] for s in self.data)
        print(f"Dataset length: {length}")
        return length

    def __getitem__(self, idx):
        print(f"Getting item at index {idx}")
        seq_idx, frame_idx = self._get_seq_frame(idx)
        seq = self.data[seq_idx]
        f = frame_idx + 1
        img_path = os.path.join(seq["img_dir"], f"{f:08d}{seq['seq_info']['imExt']}")
        print(f"Loading image from {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        boxes, ids = [], []
        if "gt" in seq:
            df = seq["gt"]
            row = df[df["frame"] == f]
            boxes = row[["x", "y", "w", "h"]].values
            ids = row["id"].values
            scale_x = self.img_size[0] / seq["seq_info"]["imWidth"]
            scale_y = self.img_size[1] / seq["seq_info"]["imHeight"]
            boxes = boxes * [scale_x, scale_y, scale_x, scale_y]
            print(f"Frame {f}: Found {len(boxes)} boxes, {len(ids)} IDs")
        sample = {
            "image": torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4)),
            "ids": torch.tensor(ids, dtype=torch.int64) if len(ids) > 0 else torch.empty((0,))
        }
        if self.transform:
            sample = self.transform(sample)
            print(f"Applied transform, new image shape: {sample['image'].shape}")
        print(f"Sample shapes - image: {sample['image'].shape}, boxes: {sample['boxes'].shape}, ids: {sample['ids'].shape}")
        return sample

    def _get_seq_frame(self, idx):
        total = 0
        for i, s in enumerate(self.data):
            length = s["seq_info"]["seqLength"]
            if total + length > idx:
                print(f"Mapping idx {idx} to seq {i}, frame {idx - total}")
                return i, idx - total
            total += length
        print(f"Mapping idx {idx} to last seq, frame {length - 1}")
        return len(self.data) - 1, length - 1