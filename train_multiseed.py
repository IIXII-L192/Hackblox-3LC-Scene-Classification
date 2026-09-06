import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.swa_utils import AveragedModel, SWALR
from PIL import Image
import tlc
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import os
import sys
import math

SEEDS = [42, 101, 777, 2024, 999]
EPOCHS = 45
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WARMUP_EPOCHS = 3
SWA_START_EPOCH = 32
SWA_LR = 0.0003
PROJECT_NAME = "Intel-Scene"
DATASET_NAME = "intel-scene"
NUM_CLASSES = 6
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street", "undefined"]
MAX_WEIGHT1_ROWS = 3000
IMAGE_SIZE = 224

MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
AUG_PROB = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        os.environ["PYTHONHASHSEED"] = str(seed)


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.2)),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def train_fn(sample):
    image = Image.open(sample["image"])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return train_transform(image), sample["label"]


def val_fn(sample):
    image = Image.open(sample["image"])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return val_transform(image), sample["label"]


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    rand_index = torch.randperm(x.size(0)).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_cut = x.clone()
    x_cut[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x_cut, y, y[rand_index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_lr_with_warmup(epoch, warmup_epochs, base_lr, total_epochs):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_single_seed(seed, train_table, val_table, val_dataloader):
    print("\n" + "=" * 60)
    print(f"  Training Seed {seed} (40 Epochs, Mixup, CutMix, SWA)")
    print("=" * 60)
    set_seed(seed)

    train_sampler = train_table.create_sampler(exclude_zero_weights=True)
    train_dataloader = DataLoader(
        train_table,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
    )

    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    best_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        if epoch < SWA_START_EPOCH:
            lr = get_lr_with_warmup(epoch, WARMUP_EPOCHS, LEARNING_RATE, SWA_START_EPOCH)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        epoch_loss = 0.0
        n_b = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            r = random.random()
            if epoch >= WARMUP_EPOCHS and r < 0.40:
                mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                optimizer.zero_grad()
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif epoch >= WARMUP_EPOCHS and r < 0.80:
                cut_images, targets_a, targets_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                optimizer.zero_grad()
                outputs = model(cut_images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        if epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                v_corr += (pred == labels).sum().item()
                v_tot += labels.size(0)
        acc = 100 * v_corr / v_tot
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Seed {seed} | Ep {epoch+1}/{EPOCHS} | Val Acc: {acc:.2f}% | Loss: {epoch_loss/n_b:.4f} | LR: {current_lr:.6f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nUpdating SWA BN for Seed {seed}...")
    torch.optim.swa_utils.update_bn(train_dataloader, swa_model, device=device)
    swa_model.eval()
    s_corr, s_tot = 0, 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            pred = swa_model(images).argmax(1)
            s_corr += (pred == labels).sum().item()
            s_tot += labels.size(0)
    swa_acc = 100 * s_corr / s_tot
    print(f"Seed {seed} SWA Val Acc: {swa_acc:.2f}% (Best Single: {best_acc:.2f}%)")

    if swa_acc > best_acc:
        final_state = {k.replace("module.", ""): v.cpu().clone() for k, v in swa_model.state_dict().items()}
        final_acc = swa_acc
    else:
        final_state = best_state
        final_acc = best_acc

    save_path = f"best_model_seed{seed}.pth"
    torch.save(final_state, save_path)
    print(f"[OK] Saved Seed {seed} checkpoint to {save_path} (Acc: {final_acc:.2f}%)")
    return final_state, final_acc


def main():
    print("=" * 70)
    print("  3LC Intel Scene - Multi-Seed Ensemble Trainer")
    print("=" * 70)

    train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train",
    ).latest()
    val_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="val",
    ).latest()

    print(f"  Train: {len(train_table)} samples ({train_table.url})")
    print(f"  Val:   {len(val_table)} samples ({val_table.url})")

    n_weight1 = sum(1 for row in train_table.table_rows if row["weight"] > 0)
    print(f"Labeling budget: {n_weight1} / {MAX_WEIGHT1_ROWS} weight-1 rows used")
    if n_weight1 > MAX_WEIGHT1_ROWS:
        print(f"[ERROR] Budget exceeded ({n_weight1} > {MAX_WEIGHT1_ROWS})")
        return 1

    train_table.map(train_fn).map_collect_metrics(val_fn)
    val_table.map(val_fn)
    val_dataloader = DataLoader(val_table, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_global_acc = 0.0
    best_global_state = None

    for seed in SEEDS:
        state, acc = train_single_seed(seed, train_table, val_table, val_dataloader)
        if acc > best_global_acc:
            best_global_acc = acc
            best_global_state = state

    torch.save(best_global_state, "best_model.pth")
    print("\n" + "=" * 70)
    print(f"  Multi-Seed Training Complete! Highest Validation Acc: {best_global_acc:.2f}%")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
