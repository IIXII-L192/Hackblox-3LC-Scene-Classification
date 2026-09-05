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
from tqdm import tqdm

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street", "undefined"]
PROJECT_NAME = "Intel-Scene"
DATASET_NAME = "intel-scene"
TARGET_PER_CLASS = 500
PROMOTIONS_PER_CLASS = 400
MAX_ALLOWED_WEIGHT1 = 3000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


tta_transforms = [
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


def resolve_image_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p
    resolved = base_dir / p
    if resolved.exists():
        return resolved
    try:
        relative_part = p.relative_to(p.anchor)
        resolved = base_dir / relative_part
        if resolved.exists():
            return resolved
    except ValueError:
        pass
    parts = p.parts
    for i in range(len(parts)):
        subpath = Path(*parts[i:])
        resolved = base_dir / subpath
        if resolved.exists():
            return resolved
    return p


def main():
    print("=" * 70)
    print("  3LC Intel-Scene - High-Confidence TTA & Margin Active Learning Curation")
    print("=" * 70)

    model_path = Path("best_model.pth")
    if not model_path.exists():
        print(f"[ERROR] {model_path} not found!")
        return 1

    print("\n[1/5] Loading trained ResNet-18 model...")
    state_dict = torch.load(model_path, map_location=device)
    model = ResNet18Classifier(num_classes=6)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  [OK] Model successfully loaded.")

    print("\n[2/5] Loading 3LC base train table (original seed dataset)...")
    try:
        train_table_url = tlc.Url.create_table_url(
            table_name="train",
            dataset_name=DATASET_NAME,
            project_name=PROJECT_NAME
        )
        base_train_table = tlc.Table.from_url(train_table_url)
    except Exception:
        base_train_table = tlc.Table.from_url(tlc.Url.create_table_url("train_0000", DATASET_NAME, PROJECT_NAME))
    
    print(f"  Loaded table: {base_train_table.url}")
    print(f"  Total samples in table: {len(base_train_table)}")

    base_dir = Path.cwd()

    print("\n[3/5] Running 8-view TTA inference & margin calculation on undefined samples...")
    undefined_candidates = []

    with torch.no_grad():
        for row_idx, row in enumerate(tqdm(base_train_table, desc="TTA Inference on Undefined")):
            raw_label = row["label"]
            label_val = raw_label if isinstance(raw_label, int) else raw_label.get("label", 6)
            
            if label_val != 6:
                continue

            img_path = resolve_image_path(str(row["image"]), base_dir)
            try:
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
            except Exception as e:
                continue

            tensors = [t(img).unsqueeze(0).to(device) for t in tta_transforms]
            batch_tensors = torch.cat(tensors, dim=0)
            logits = model(batch_tensors)
            probs = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()

            sorted_indices = np.argsort(probs)[::-1]
            top1_cls = int(sorted_indices[0])
            top2_cls = int(sorted_indices[1])
            top1_prob = float(probs[top1_cls])
            top2_prob = float(probs[top2_cls])
            margin = top1_prob - top2_prob

            undefined_candidates.append({
                "row_idx": row_idx,
                "predicted_label": top1_cls,
                "top1_prob": top1_prob,
                "top2_cls": top2_cls,
                "top2_prob": top2_prob,
                "margin": margin,
            })

    df_cands = pd.DataFrame(undefined_candidates)
    print(f"\n  Found {len(df_cands)} undefined candidates.")

    print("\n[4/5] Selecting Top 400 highest-margin clean samples per class...")
    selected_promotions = {}
    
    for c in range(6):
        cls_df = df_cands[df_cands["predicted_label"] == c].sort_values(by="margin", ascending=False)
        print(f"\n  Class {c} ({CLASSES[c]}): {len(cls_df)} candidates available")
        if len(cls_df) < PROMOTIONS_PER_CLASS:
            print(f"    WARNING: Only {len(cls_df)} candidates available for class {c}")
            top_c = cls_df
        else:
            top_c = cls_df.head(PROMOTIONS_PER_CLASS)
            
        avg_conf = top_c["top1_prob"].mean()
        avg_margin = top_c["margin"].mean()
        min_conf = top_c["top1_prob"].min()
        print(f"    Selected 400 | Avg Conf: {avg_conf:.4f} | Avg Margin: {avg_margin:.4f} | Min Conf: {min_conf:.4f}")

        for _, r in top_c.iterrows():
            selected_promotions[int(r["row_idx"])] = int(r["predicted_label"])

    print(f"\n  Total promotions selected: {len(selected_promotions)}")

    print("\n[5/5] Creating new balanced 3LC table revision (3,000 weight-1 rows)...")
    new_rows = []
    weight1_counts = {c: 0 for c in range(7)}

    for row_idx, row in enumerate(base_train_table):
        new_row = dict(row)
        raw_label = row["label"]
        current_label = raw_label if isinstance(raw_label, int) else raw_label.get("label", 6)

        if row_idx in selected_promotions:
            promoted_label = selected_promotions[row_idx]
            new_row["label"] = promoted_label
            new_row["weight"] = 1.0
            weight1_counts[promoted_label] += 1
        elif current_label < 6:
            new_row["weight"] = 1.0
            weight1_counts[current_label] += 1
        else:
            new_row["label"] = 6
            new_row["weight"] = 0.0
            weight1_counts[6] += 1

        new_rows.append(new_row)

    total_weight1 = sum(v for k, v in weight1_counts.items() if k < 6)
    print("\n  Summary of Active (Weight=1) Samples per Class in New Revision:")
    for c in range(6):
        print(f"    {c} ({CLASSES[c]}): {weight1_counts[c]} samples")
    print(f"    Total Active Samples: {total_weight1} / {MAX_ALLOWED_WEIGHT1} (Budget limit)")

    if total_weight1 > MAX_ALLOWED_WEIGHT1:
        print(f"  [ERROR] Exceeded maximum allowed weight-1 rows ({total_weight1} > {MAX_ALLOWED_WEIGHT1})")
        return 1

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
        description=f"Advanced TTA Margin Curation: Exactly 500 balanced samples/class = {total_weight1} total",
        column_schemas=schemas,
        input_tables=[base_train_table.url],
    )

    for r in new_rows:
        writer.add_row({
            "id": int(r["id"]),
            "image": str(r["image"]),
            "label": int(r["label"]) if isinstance(r["label"], int) else int(r["label"].get("label", 6)),
            "weight": float(r["weight"]),
        })

    new_table = writer.finalize()
    print(f"\n  [SUCCESS] Created new table revision: {new_table.url}")
    print("=" * 70)
    print("  Active learning curation complete! Ready for train.py.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
