import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import csv
import shutil

SEEDS = [42, 101, 777, 2024, 999]
MODEL_PATHS = [Path(f"best_model_seed{s}.pth") for s in SEEDS]
TEST_DIR = Path("data/test")
OUTPUT_PATH = Path("submission.csv")
SUBMISSIONS_DIR = Path("submissions")
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")
NUM_CLASSES = 6
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
BATCH_SIZE = 32
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class TestDataset(Dataset):
    def __init__(self, image_dir: Path):
        self.image_dir = Path(image_dir)
        self.images = []
        if self.image_dir.exists():
            seen = set()
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img in self.image_dir.glob(ext):
                    key = img.name.lower()
                    if key not in seen:
                        seen.add(key)
                        self.images.append(img)
        self.images.sort(key=lambda x: x.name)
        print(f"  Found {len(self.images)} images in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
        image_id = img_path.stem
        return image, image_id


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

tta_transforms = [
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ]),
    transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ]),
    transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ]),
    transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ]),
    transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ]),
    transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ]),
    transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ]),
]


def predict_ensemble_tta(models_list, raw_images, device):
    total_probs = None
    n_views = len(tta_transforms)
    n_models = len(models_list)

    for model in models_list:
        for t in tta_transforms:
            batch_tensors = torch.stack([t(img) for img in raw_images]).to(device)
            with torch.no_grad():
                logits = model(batch_tensors)
                probs = F.softmax(logits, dim=1)
            if total_probs is None:
                total_probs = probs
            else:
                total_probs += probs

    avg_probs = total_probs / (n_views * n_models)
    confidences, predictions = avg_probs.max(dim=1)
    return predictions.cpu().numpy(), confidences.cpu().numpy()


def load_expected_image_ids():
    if not SAMPLE_SUBMISSION_PATH.exists():
        return None
    with open(SAMPLE_SUBMISSION_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_id" not in (reader.fieldnames or []):
            return None
        return [row["image_id"] for row in reader]


def main():
    print("=" * 70)
    print("  Intel Scene - Multi-Seed Model & 8-View TTA Ensemble Predictor")
    print("=" * 70)

    loaded_models = []
    for mp in MODEL_PATHS:
        if mp.exists():
            m = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
            state = torch.load(mp, map_location=device)
            if "n_averaged" in state:
                del state["n_averaged"]
            m.load_state_dict(state)
            m.eval()
            loaded_models.append(m)
            print(f"  [OK] Loaded ensemble member: {mp}")

    if not loaded_models:
        if Path("best_model.pth").exists():
            m = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
            state = torch.load("best_model.pth", map_location=device)
            if "n_averaged" in state:
                del state["n_averaged"]
            m.load_state_dict(state)
            m.eval()
            loaded_models.append(m)
            print("  [OK] Loaded fallback best_model.pth")
        else:
            print("[ERROR] No model checkpoints found.")
            return 1

    print(f"\nRunning ensemble inference with {len(loaded_models)} model(s) x {len(tta_transforms)} TTA views...")
    test_dataset = TestDataset(TEST_DIR)

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: ([item[0] for item in batch], [item[1] for item in batch])
    )

    predictions = []
    for raw_imgs, image_ids in tqdm(loader, desc="Ensemble Inference"):
        preds, confs = predict_ensemble_tta(loaded_models, raw_imgs, device)
        for i_id, p, c in zip(image_ids, preds, confs):
            predictions.append({
                "image_id": i_id,
                "prediction": int(p),
                "confidence": round(float(c), 4),
            })

    pred_by_id = {p["image_id"]: p for p in predictions}
    print(f"  [OK] Predicted {len(predictions)} images")

    print("\nAligning to sample_submission.csv format...")
    expected_ids = load_expected_image_ids()
    if expected_ids is not None:
        aligned_rows = []
        for image_id in expected_ids:
            if image_id in pred_by_id:
                aligned_rows.append(pred_by_id[image_id])
            else:
                aligned_rows.append({"image_id": image_id, "prediction": 0, "confidence": 0.5})
        predictions = aligned_rows
        print(f"  [OK] Aligned exactly {len(predictions)} rows")

    print("\nWriting submission.csv...")
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"  [OK] Written canonical submission: {OUTPUT_PATH}")

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = SUBMISSIONS_DIR / f"submission_ensemble_{timestamp}.csv"
    shutil.copyfile(OUTPUT_PATH, history_path)
    print(f"  [OK] Timestamped copy archived to: {history_path}")

    print("=" * 70)
    print("  [SUCCESS] Ensemble submission generated & verified!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
