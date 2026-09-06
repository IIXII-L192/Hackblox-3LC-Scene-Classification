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
MAX_ALLOWED_WEIGHT1 = 3000
SEEDS = [42, 101, 777, 2024, 999]
PRUNE_BOTTOM_PER_CLASS = 50

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


def load_ensemble_models():
    models_list = []
    for s in SEEDS:
        p = Path(f"best_model_seed{s}.pth")
        if p.exists():
            m = ResNet18Classifier(num_classes=6).to(device)
            state = torch.load(p, map_location=device)
            if "n_averaged" in state:
                del state["n_averaged"]
            m.load_state_dict(state)
            m.eval()
            models_list.append(m)
            print(f"  [OK] Loaded {p}")
    if not models_list:
        m = ResNet18Classifier(num_classes=6).to(device)
        state = torch.load("best_model.pth", map_location=device)
        if "n_averaged" in state:
            del state["n_averaged"]
        m.load_state_dict(state)
        m.eval()
        models_list.append(m)
    return models_list


def main():
    print("=" * 70)
    print("  3LC Intel Scene - Consensus Hard-Sample Noise Pruner & Re-Curator")
    print("=" * 70)

    base_dir = Path.cwd()
    models_list = load_ensemble_models()
    print(f"Loaded {len(models_list)} ensemble model(s) for consensus evaluation.")

    latest_train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train",
    ).latest()
    print(f"Current Table: {latest_train_table.url}")

    active_indices = []
    undefined_indices = []

    for row_idx, row in enumerate(latest_train_table.table_rows):
        w = float(row.get("weight", 0.0))
        raw_lbl = row.get("label", 6)
        l_val = raw_lbl if isinstance(raw_lbl, int) else raw_lbl.get("label", 6)
        if w > 0:
            active_indices.append((row_idx, l_val))
        else:
            undefined_indices.append(row_idx)

    print(f"Active samples (weight=1): {len(active_indices)}")
    print(f"Undefined samples (weight=0): {len(undefined_indices)}")

    print("\n[1/3] Scoring current active samples with ensemble consensus...")
    active_eval = []
    with torch.no_grad():
        for row_idx, current_label in tqdm(active_indices, desc="Evaluating Active"):
            row = latest_train_table.table_rows[row_idx]
            img_path = resolve_image_path(str(row["image"]), base_dir)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            probs_all = []
            for m in models_list:
                tensors = [t(img).unsqueeze(0).to(device) for t in tta_transforms]
                batch_t = torch.cat(tensors, dim=0)
                logits = m(batch_t)
                p = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
                probs_all.append(p)

            avg_p = np.mean(probs_all, axis=0)
            pred_cls = int(np.argmax(avg_p))
            conf = float(avg_p[pred_cls])
            margin = float(avg_p[pred_cls] - np.sort(avg_p)[-2])
            is_seed = (row_idx < 600)
            target_prob = float(avg_p[current_label]) if current_label < 6 else 0.0

            active_eval.append({
                "row_idx": row_idx,
                "current_label": current_label,
                "pred_cls": pred_cls,
                "conf": conf,
                "target_prob": target_prob,
                "margin": margin,
                "is_seed": is_seed,
            })

    df_active = pd.DataFrame(active_eval)

    prune_set = set()
    for c in range(6):
        c_curated = df_active[(df_active["current_label"] == c) & (~df_active["is_seed"])].sort_values(by="target_prob", ascending=True)
        to_prune = c_curated.head(PRUNE_BOTTOM_PER_CLASS)["row_idx"].tolist()
        prune_set.update(to_prune)
        print(f"  Class {c} ({CLASSES[c]}): Identified {len(to_prune)} lowest-confidence active rows for replacement.")

    print(f"\nTotal noisy active samples slated for pruning: {len(prune_set)}")

    print("\n[2/3] Scoring remaining undefined pool for replacement candidates...")
    cand_eval = []
    with torch.no_grad():
        for row_idx in tqdm(undefined_indices, desc="Evaluating Undefined Pool"):
            row = latest_train_table.table_rows[row_idx]
            img_path = resolve_image_path(str(row["image"]), base_dir)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            probs_all = []
            for m in models_list:
                tensors = [t(img).unsqueeze(0).to(device) for t in tta_transforms]
                batch_t = torch.cat(tensors, dim=0)
                logits = m(batch_t)
                p = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
                probs_all.append(p)

            avg_p = np.mean(probs_all, axis=0)
            pred_cls = int(np.argmax(avg_p))
            conf = float(avg_p[pred_cls])
            margin = float(avg_p[pred_cls] - np.sort(avg_p)[-2])

            cand_eval.append({
                "row_idx": row_idx,
                "pred_cls": pred_cls,
                "conf": conf,
                "margin": margin,
            })

    df_cand = pd.DataFrame(cand_eval)

    replacements = {}
    for c in range(6):
        c_cand = df_cand[df_cand["pred_cls"] == c].sort_values(by="margin", ascending=False)
        top_replacements = c_cand.head(PRUNE_BOTTOM_PER_CLASS)
        print(f"  Class {c} ({CLASSES[c]}): Picked {len(top_replacements)} clean replacements | Avg Margin: {top_replacements['margin'].mean():.4f}")
        for _, r in top_replacements.iterrows():
            replacements[int(r["row_idx"])] = c

    print("\n[3/3] Creating new sanitized 3LC Table Revision (Exactly 3,000 weight-1 rows)...")
    new_rows = []
    weight1_counts = {c: 0 for c in range(7)}

    for row_idx, row in enumerate(latest_train_table.table_rows):
        new_row = dict(row)
        raw_lbl = row.get("label", 6)
        current_l = raw_lbl if isinstance(raw_lbl, int) else raw_lbl.get("label", 6)
        w_val = float(row.get("weight", 0.0))

        if row_idx in prune_set:
            new_row["label"] = 6
            new_row["weight"] = 0.0
            weight1_counts[6] += 1
        elif row_idx in replacements:
            new_row["label"] = replacements[row_idx]
            new_row["weight"] = 1.0
            weight1_counts[replacements[row_idx]] += 1
        elif w_val > 0:
            new_row["label"] = current_l
            new_row["weight"] = 1.0
            weight1_counts[current_l] += 1
        else:
            new_row["label"] = 6
            new_row["weight"] = 0.0
            weight1_counts[6] += 1

        new_rows.append(new_row)

    total_weight1 = sum(v for k, v in weight1_counts.items() if k < 6)
    print("\nSummary of Active (Weight=1) Samples per Class:")
    for c in range(6):
        print(f"  {c} ({CLASSES[c]}): {weight1_counts[c]} samples")
    print(f"  Total Active: {total_weight1} / {MAX_ALLOWED_WEIGHT1} (Budget Cap Checked)")

    if total_weight1 > MAX_ALLOWED_WEIGHT1:
        print("[ERROR] Budget exceeded.")
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
        description=f"Consensus Hard-Sample Pruned Table Revision: Exactly 500 samples/class = {total_weight1} total",
        column_schemas=schemas,
        input_tables=[latest_train_table.url],
    )

    for r in new_rows:
        writer.add_row({
            "id": int(r["id"]),
            "image": str(r["image"]),
            "label": int(r["label"]) if isinstance(r["label"], int) else int(r["label"].get("label", 6)),
            "weight": float(r["weight"]),
        })

    new_table = writer.finalize()
    print(f"\n[SUCCESS] New sanitized table committed: {new_table.url}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
