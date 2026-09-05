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
TARGET_PER_CLASS = 475
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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

flip_transform = transforms.Compose([
    transforms.Resize(256),
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


def create_balanced_curated_revision():
    base_dir = Path(__file__).parent.absolute()
    print("=" * 70)
    print("  Creating Balanced 3LC Table Revision (Target: 475 / class = 2,850 total)")
    print("=" * 70)

    init_train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train",
    )
    df = init_train_table.to_pandas()
    
    df["weight"] = 0.0
    for i in range(600):
        df.loc[i, "weight"] = 1.0

    print(f"[OK] Base seed labeled samples: 600 (100 per class)")

    model = ResNet18Classifier(num_classes=6)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    pool_indices = list(range(600, 6600))
    print(f"Evaluating pool of {len(pool_indices)} undefined images with TTA...")

    predictions_data = []
    with torch.no_grad():
        for idx in tqdm(pool_indices):
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
                "probs": avg_probs.cpu().numpy()[0],
            })

    pred_df = pd.DataFrame(predictions_data)

    new_df = df.copy()
    promoted_per_class = {}

    for c_idx in range(6):
        pred_df[f"prob_{c_idx}"] = pred_df["probs"].apply(lambda p: p[c_idx])
        c_sorted = pred_df.sort_values(by=f"prob_{c_idx}", ascending=False)
        
        top_candidates = []
        for _, row in c_sorted.iterrows():
            idx = int(row["df_index"])
            if new_df.loc[idx, "weight"] == 0.0:
                top_candidates.append(idx)
                if len(top_candidates) == (TARGET_PER_CLASS - 100):
                    break
        
        for idx in top_candidates:
            new_df.loc[idx, "label"] = c_idx
            new_df.loc[idx, "weight"] = 1.0
            
        promoted_per_class[CLASSES[c_idx]] = len(top_candidates) + 100
        print(f"  Class {c_idx} ({CLASSES[c_idx]}): 100 seed + {len(top_candidates)} curated = {len(top_candidates) + 100} total")

    total_weight1 = int((new_df["weight"] > 0).sum())
    print("\n" + "=" * 70)
    print(f"Total Active Training Samples: {total_weight1} / {MAX_ALLOWED_WEIGHT1} (STRICTLY BALANCED)")
    print("=" * 70)

    schemas = {
        "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
        "image": tlc.ImagePath,
        "label": tlc.CategoricalLabel("label", classes=CLASSES),
        "weight": tlc.SampleWeightSchema(),
    }

    latest_train = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train",
    ).latest()

    writer = tlc.TableWriter(
        table_name="train",
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        description=f"Balanced Active Curation: Exactly 475 samples per class (Total: {total_weight1})",
        column_schemas=schemas,
        input_tables=[latest_train.url],
    )

    for _, r in new_df.iterrows():
        writer.add_row({
            "id": int(r["id"]),
            "image": str(r["image"]),
            "label": int(r["label"]),
            "weight": float(r["weight"]),
        })

    new_table = writer.finalize()
    print(f"\n[OK] Balanced Revision committed: {new_table.url}")


if __name__ == "__main__":
    create_balanced_curated_revision()
