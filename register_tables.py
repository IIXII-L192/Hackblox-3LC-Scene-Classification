import tlc
from pathlib import Path

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street", "undefined"]
PROJECT_NAME = "Intel-Scene"
DATASET_NAME = "intel-scene"

schemas = {
    "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
    "image": tlc.ImagePath,
    "label": tlc.CategoricalLabel("label", classes=CLASSES),
    "weight": tlc.SampleWeightSchema(),
}


def register_dataset_to_table(
    dataset_path: Path,
    table_name: str,
    split_name: str,
    include_undefined: bool = False,
):
    dataset_path = Path(dataset_path)
    image_data = []

    classes_to_process = CLASSES[:-1] if not include_undefined else CLASSES

    for class_idx, class_name in enumerate(classes_to_process):
        class_folder = dataset_path / class_name
        if class_folder.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_files = sorted(class_folder.glob(ext))
                if image_files:
                    print(f"  Found {len(image_files):5d} images in {class_name}/ for {split_name}")
                for img_path in image_files:
                    label = class_idx if class_name != "undefined" else 6
                    image_data.append({"path": str(img_path.absolute()), "label": label})
        else:
            if class_name != "undefined":
                print(f"  [WARN] {class_folder} does not exist")

    print(f"\n  Total images for {split_name}: {len(image_data)}")

    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        description=f"Intel Scene {split_name} set with {len(image_data)} images",
        column_schemas=schemas,
        if_exists="overwrite",
    )

    for i, data in enumerate(image_data):
        label = data["label"]
        weight = 1.0 if label in range(6) else 0.0
        table_writer.add_row({
            "id": i,
            "image": data["path"],
            "label": label,
            "weight": weight,
        })

    table = table_writer.finalize()
    num_labeled = sum(1 for x in image_data if x["label"] in range(6))
    num_undefined = sum(1 for x in image_data if x["label"] == 6)
    print(f"\n  [OK] Created 3LC table: '{table_name}'")
    print(f"  Labeled (weight=1.0): {num_labeled}, Undefined (weight=0.0): {num_undefined}")
    print(f"  Table URL: {table.url}")
    return table


def tables_exist():
    try:
        train_ref = tlc.Table.from_names(
            project_name=PROJECT_NAME,
            dataset_name=DATASET_NAME,
            table_name="train",
        )
        val_ref = tlc.Table.from_names(
            project_name=PROJECT_NAME,
            dataset_name=DATASET_NAME,
            table_name="val",
        )
        return True, train_ref, val_ref
    except Exception:
        return False, None, None


def main():
    base_path = Path(__file__).parent
    data_path = base_path / "data"

    print("=" * 70)
    print("  Registering Intel Scene Dataset in 3LC Tables")
    print("=" * 70)

    if not data_path.exists():
        print(f"\n[ERROR] Data directory not found: {data_path}")
        return

    exist, train_ref, val_ref = tables_exist()
    if exist:
        print("\n  [IDEMPOTENT] Train and val tables already exist. Skipping registration.")
        if train_ref is not None and val_ref is not None:
            try:
                train_table = train_ref.latest()
                val_table = val_ref.latest()
                print(f"  Train table URL: {train_table.url}")
                print(f"  Val table URL:   {val_table.url}")
            except Exception:
                pass
        return

    print("\nRegistering URL alias for portable paths...")
    tlc.register_project_url_alias(
        token="INTEL_SCENE_DATA",
        path=str(base_path.absolute()),
        project=PROJECT_NAME,
    )
    print(f"  [OK] Alias registered -> {base_path.absolute()}")

    print("\n" + "-" * 70)
    print("[1/2] Registering TRAIN set (includes undefined)...")
    print("-" * 70)
    train_table = register_dataset_to_table(
        data_path / "train",
        table_name="train",
        split_name="train",
        include_undefined=True,
    )

    print("\n" + "-" * 70)
    print("[2/2] Registering VAL set...")
    print("-" * 70)
    val_table = register_dataset_to_table(
        data_path / "val",
        table_name="val",
        split_name="val",
        include_undefined=False,
    )

    print("\n" + "=" * 70)
    print("  [OK] Successfully registered tables!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
