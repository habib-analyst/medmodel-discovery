"""MedMNIST data loading from local npz files (downloaded via curl)."""
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = os.environ.get("MMD_ROOT", "/home/hatch/.medmnist")


class MedMNISTDataset(Dataset):
    def __init__(self, name, split, repeat_gray=True):
        d = np.load(f"{ROOT}/{name}.npz")
        imgs = d[f"{split}_images"]
        labels = d[f"{split}_labels"].reshape(-1)
        if imgs.ndim == 3:  # grayscale -> (H,W) to (H,W,1)
            imgs = imgs[..., None]
        self.imgs = imgs.astype(np.float32) / 255.0
        self.labels = labels.astype(np.int64)
        self.repeat_gray = repeat_gray
        self.n_classes = int(labels.max()) + 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.imgs[i]  # H,W,C
        if self.repeat_gray and x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        x = torch.from_numpy(x).permute(2, 0, 1)  # C,H,W
        return x, torch.tensor(self.labels[i])


def get_loaders(name, batch=128, subsample=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    train = MedMNISTDataset(name, "train")
    if subsample and len(train) > subsample:
        idx = torch.randperm(len(train), generator=g)[:subsample]
        train = torch.utils.data.Subset(train, idx)
    val = MedMNISTDataset(name, "val")
    test = MedMNISTDataset(name, "test")
    kw = dict(batch_size=batch, num_workers=0)
    return (DataLoader(train, shuffle=True, generator=g, **kw),
            DataLoader(val, shuffle=False, **kw),
            DataLoader(test, shuffle=False, **kw),
            train.n_classes if not isinstance(train, torch.utils.data.Subset)
            else train.dataset.n_classes)
