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

EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WARMUP_EPOCHS = 3
SWA_START_EPOCH = 30
SWA_LR = 0.0003
RANDOM_SEED = 42
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
print(f"ResNet-18: STANDARD architecture, random init (no pretrained weights — competition rules)")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE} (ResNet-18 native resolution)")


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[OK] Random seed set to {seed}")


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


def metrics_fn(batch, predictor_output: tlc.PredictorOutput):
    labels = batch[1].to(device)
    predictions = predictor_output.forward
    softmax_output = F.softmax(predictions, dim=1)
    predicted_indices = torch.argmax(predictions, dim=1)
    confidence = torch.gather(softmax_output, 1, predicted_indices.unsqueeze(1)).squeeze(1)
    accuracy = (predicted_indices == labels).float()
    valid_labels = labels < predictions.shape[1]
    cross_entropy_loss = torch.ones_like(labels, dtype=torch.float32)
    cross_entropy_loss[valid_labels] = nn.CrossEntropyLoss(reduction="none")(
        predictions[valid_labels], labels[valid_labels]
    )
    return {
        "loss": cross_entropy_loss.cpu().numpy(),
        "predicted": predicted_indices.cpu().numpy(),
        "accuracy": accuracy.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }


def get_lr_with_warmup(epoch, warmup_epochs, base_lr, total_epochs):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


BEST_MODEL_FILENAME = "best_model.pth"


def train():
    set_seed(RANDOM_SEED)
    base_path = Path(__file__).parent
    tlc.register_project_url_alias(
        token="INTEL_SCENE_DATA",
        path=str(base_path.absolute()),
        project=PROJECT_NAME,
    )
    print(f"[OK] Registered data path: {base_path.absolute()}")
    print("\nLoading 3LC tables...")

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

    print(f"  Train: {len(train_table)} samples")
    print(f"  Val:   {len(val_table)} samples")
    print(f"  Train table URL: {train_table.url}")
    print(f"  Val table URL:   {val_table.url}")
    class_names = list(train_table.get_simple_value_map("label").values())
    print(f"  Classes: {class_names}")

    n_weight1 = sum(1 for row in train_table.table_rows if row["weight"] > 0)
    print(f"Labeling budget: {n_weight1} / {MAX_WEIGHT1_ROWS} weight-1 rows used")
    if n_weight1 > MAX_WEIGHT1_ROWS:
        print("\n" + "=" * 60)
        print("  [ERROR] Labeling budget exceeded")
        print("=" * 60)
        print(f"  This train table revision has {n_weight1} rows with weight = 1.")
        print(f"  The competition rule allows at most {MAX_WEIGHT1_ROWS} weight-1 rows in the")
        print("  final train table (the 600 seed labels count toward this).")
        sys.exit(1)

    train_table.map(train_fn).map_collect_metrics(val_fn)
    val_table.map(val_fn)
    train_sampler = train_table.create_sampler(exclude_zero_weights=True)
    train_dataloader = DataLoader(
        train_table,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
    )
    val_dataloader = DataLoader(val_table, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    run = tlc.init(
        project_name=PROJECT_NAME,
        description=f"Intel Scene - standard ResNet-18, {IMAGE_SIZE}px, Mixup, SWA",
    )
    metric_schemas = {
        "loss": tlc.Schema(description="Cross entropy loss", value=tlc.Float32Value()),
        "predicted": tlc.CategoricalLabelSchema(display_name="predicted label", classes=class_names),
        "accuracy": tlc.Schema(description="Per-sample accuracy", value=tlc.Float32Value()),
        "confidence": tlc.Schema(description="Prediction confidence", value=tlc.Float32Value()),
    }
    classification_metrics_collector = tlc.FunctionalMetricsCollector(
        collection_fn=metrics_fn,
        column_schemas=metric_schemas,
    )
    indices_and_modules = list(enumerate(model.resnet.named_modules()))
    resnet_fc_layer_index = next((i for i, (n, _) in indices_and_modules if n == "fc"), len(indices_and_modules) - 1)
    embeddings_metrics_collector = tlc.EmbeddingsMetricsCollector(layers=[resnet_fc_layer_index])
    predictor = tlc.Predictor(model, layers=[resnet_fc_layer_index])

    best_val_accuracy = 0.0
    best_model_state = None
    print("\n" + "=" * 60)
    print(f"  Starting Training (Phase 2 - 40 Epochs with Mixup & CutMix)")
    print(f"  Epochs: {EPOCHS} | Image: {IMAGE_SIZE}px | Aug Prob: {AUG_PROB}")
    print(f"  LR: {LEARNING_RATE} (warmup {WARMUP_EPOCHS}ep) | SWA from epoch {SWA_START_EPOCH}")
    print(f"  Model: STANDARD ResNet-18 (rules-compliant)")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()

        if epoch < SWA_START_EPOCH:
            lr = get_lr_with_warmup(epoch, WARMUP_EPOCHS, LEARNING_RATE, SWA_START_EPOCH)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        epoch_loss = 0.0
        n_batches = 0

        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
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
            n_batches += 1

        if epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        val_accuracy = 100 * val_correct / val_total
        avg_loss = epoch_loss / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} - Val Acc: {val_accuracy:.2f}% - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  --> New best model!")
        tlc.log({"epoch": epoch, "val_accuracy": val_accuracy, "train_loss": avg_loss})

    print("\n" + "=" * 60)
    print(f"  Best single-model val accuracy: {best_val_accuracy:.2f}%")
    print("=" * 60)

    print("\nUpdating SWA batch normalization statistics...")
    try:
        torch.optim.swa_utils.update_bn(train_dataloader, swa_model, device=device)

        swa_model.eval()
        swa_correct, swa_total = 0, 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                pred = swa_model(images).argmax(1)
                swa_correct += (pred == labels).sum().item()
                swa_total += labels.size(0)
        swa_accuracy = 100 * swa_correct / swa_total
        print(f"SWA model val accuracy: {swa_accuracy:.2f}%")

        if swa_accuracy > best_val_accuracy:
            print(f"  --> SWA model is better! Using SWA weights.")
            final_state = {k.replace("module.", ""): v for k, v in swa_model.state_dict().items()}
            best_val_accuracy = swa_accuracy
        else:
            print(f"  --> Best single-epoch checkpoint is better. Keeping it.")
            final_state = best_model_state
    except Exception as e:
        print(f"  SWA update failed ({e}). Using best single-epoch checkpoint.")
        final_state = best_model_state

    if final_state is not None:
        model.load_state_dict(final_state)
    model_path = base_path / BEST_MODEL_FILENAME
    torch.save(model.state_dict(), model_path)
    print(f"\n[OK] Best model saved to {model_path} (overwrites previous run)")

    print("\nCollecting metrics on train set...")
    model.eval()
    tlc.collect_metrics(
        train_table,
        predictor=predictor,
        metrics_collectors=[classification_metrics_collector, embeddings_metrics_collector],
        split="train",
        dataloader_args={"batch_size": BATCH_SIZE, "num_workers": 0},
    )
    print("\nReducing embeddings...")
    try:
        run.reduce_embeddings_by_foreign_table_url(
            train_table.url,
            method="umap",
            n_neighbors=15,
            n_components=3,
        )
        print("  [OK] Embeddings reduced (UMAP, 3D).")
    except Exception as e:
        print(f"  WARNING: Embedding reduction failed: {e}")
    run.set_status_completed()
    print(f"\n[OK] Done. Final validation accuracy: {best_val_accuracy:.2f}%")
    print("[OK] View results at 3LC Dashboard (run: 3lc service)")


if __name__ == "__main__":
    train()
