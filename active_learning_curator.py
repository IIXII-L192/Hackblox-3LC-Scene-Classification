import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tlc
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street", "undefined"]
PROJECT_NAME = "Intel-Scene"
DATASET_NAME = "intel-scene"
MAX_ALLOWED_WEIGHT1 = 3000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


eval_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

flip_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def resolve_image_path(raw_path: str, base_dir: Path) -> Path:
    cleaned = raw_path.replace("<INTEL_SCENE_DATA>/", "").replace("<INTEL_SCENE_DATA>\\", "")
    p = base_dir / cleaned
    if p.exists():
        return p
    alt = Path(raw_path)
    if alt.exists():
        return alt
    return p


def curate_active_batch(batch_per_class=250, min_confidence=0.85):
    base_dir = Path(__file__).parent.absolute()
    print("=" * 70)
    print("  3LC Data-Centric Active Learning & Table Revision Creator")
    print("=" * 70)

    train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train",
    ).latest()
    print(f"[OK] Loaded train table: {train_table.url}")

    df = train_table.to_pandas()
    current_labeled_count = (df["weight"] > 0).sum()
    print(f"Current active labeled samples: {current_labeled_count} / {MAX_ALLOWED_WEIGHT1}")

    model = ResNet18Classifier(num_classes=6)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    print(f"[OK] Loaded best_model.pth on {device}")

    undefined_indices = df[df["weight"] == 0.0].index.tolist()
    print(f"Found {len(undefined_indices)} undefined samples to analyze.")

    predictions_data = []

    print("\nRunning TTA inference on undefined pool...")
    with torch.no_grad():
        for idx in tqdm(undefined_indices):
            raw_path = str(df.loc[idx, "image"])
            img_path = resolve_image_path(raw_path, base_dir)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            t1 = eval_transform(img).unsqueeze(0).to(device)
            t2 = flip_transform(img).unsqueeze(0).to(device)

            out1 = F.softmax(model(t1), dim=1)
            out2 = F.softmax(model(t2), dim=1)
            avg_probs = (out1 + out2) / 2.0

            conf, pred = avg_probs.max(1)

            predictions_data.append({
                "df_index": idx,
                "pred_label": int(pred.item()),
                "confidence": float(conf.item()),
            })

    pred_df = pd.DataFrame(predictions_data)
    print(f"\nSuccessfully evaluated {len(pred_df)} images.")
    print("Candidate predictions summary:")
    for c_idx in range(6):
        c_samples = pred_df[(pred_df["pred_label"] == c_idx) & (pred_df["confidence"] >= min_confidence)]
        print(f"  Class {c_idx} ({CLASSES[c_idx]}): {len(c_samples)} high-confidence candidates (>= {min_confidence:.2f})")

    selected_indices_to_label = []
    for c_idx in range(6):
        c_subset = pred_df[(pred_df["pred_label"] == c_idx) & (pred_df["confidence"] >= min_confidence)]
        c_sorted = c_subset.sort_values(by="confidence", ascending=False)
        top_c = c_sorted.head(batch_per_class)
        for _, row in top_c.iterrows():
            selected_indices_to_label.append((int(row["df_index"]), c_idx))

    print(f"\nTotal new samples selected for activation: {len(selected_indices_to_label)}")
    new_total_weight1 = current_labeled_count + len(selected_indices_to_label)
    print(f"Projected total weight=1 samples: {new_total_weight1} / {MAX_ALLOWED_WEIGHT1}")

    if new_total_weight1 > MAX_ALLOWED_WEIGHT1:
        excess = new_total_weight1 - MAX_ALLOWED_WEIGHT1
        print(f"[WARN] Trimming {excess} samples to strictly obey the 3,000 budget rule.")
        selected_indices_to_label = selected_indices_to_label[:-excess]

    new_df = df.copy()
    for df_idx, assigned_class in selected_indices_to_label:
        new_df.loc[df_idx, "label"] = assigned_class
        new_df.loc[df_idx, "weight"] = 1.0

    print("\nCreating new 3LC Table Revision with lineage...")
    schemas = {
        "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
        "image": tlc.ImagePath,
        "label": tlc.CategoricalLabel("label", classes=CLASSES),
        "weight": tlc.SampleWeightSchema(),
    }

    writer = tlc.TableWriter(
        table_name="train",
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        description=f"Active Learning Curation: +{len(selected_indices_to_label)} samples (total weight=1: {new_total_weight1})",
        column_schemas=schemas,
        input_tables=[train_table.url],
    )

    for _, r in new_df.iterrows():
        writer.add_row({
            "id": int(r["id"]),
            "image": str(r["image"]),
            "label": int(r["label"]),
            "weight": float(r["weight"]),
        })

    new_table = writer.finalize()
    print("=" * 70)
    print(f"  [OK] New Table Revision successfully committed!")
    print(f"  Revision URL: {new_table.url}")
    print(f"  Active Samples (weight=1.0): {(new_df['weight'] > 0).sum()} / {MAX_ALLOWED_WEIGHT1}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_per_class", type=int, default=200)
    parser.add_argument("--min_confidence", type=float, default=0.85)
    args = parser.parse_args()
    curate_active_batch(batch_per_class=args.batch_per_class, min_confidence=args.min_confidence)
