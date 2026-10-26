from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(r"C:\Users\kumar\Desktop\VISION\dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
class_counts = Counter()
invalid_labels = []
yaml_files = []

print(f"\nDataset root: {ROOT}")

if not ROOT.is_dir():
    raise SystemExit(f"Dataset directory not found: {ROOT}")

# Inspect dataset configuration files.
for path in ROOT.rglob("*.yaml"):
    yaml_files.append(path)
    print(f"\n--- YAML: {path.relative_to(ROOT)} ---")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if isinstance(data, dict):
            print("names:", data.get("names", "<missing>"))
            print("nc:", data.get("nc", "<missing>"))
            print("train:", data.get("train", "<missing>"))
            print("val:", data.get("val", "<missing>"))
            print("test:", data.get("test", "<missing>"))
        else:
            print("Unexpected YAML structure:", type(data).__name__)

    except Exception as exc:
        print("Could not read YAML:", exc)

# Count images and inspect text annotation files.
image_count = 0
label_count = 0
empty_label_count = 0

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        image_count += 1

    if path.suffix.lower() != ".txt":
        continue

    # Avoid treating unrelated text files as labels.
    if path.parent.name.lower() != "labels":
        continue

    label_count += 1

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            empty_label_count += 1

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if not parts:
                continue

            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]

                if len(values) != 4:
                    raise ValueError(
                        f"Expected 4 box coordinates, found {len(values)}"
                    )

                if not all(0 <= value <= 1 for value in values):
                    raise ValueError("Coordinates outside [0, 1]")

                if values[2] <= 0 or values[3] <= 0:
                    raise ValueError("Non-positive box width or height")

                class_counts[class_id] += 1

            except Exception as exc:
                invalid_labels.append(
                    f"{path}:{line_number}: {exc} | {line}"
                )

    except Exception as exc:
        invalid_labels.append(f"{path}: Could not read file: {exc}")

print("\n========== SUMMARY ==========")
print("Images:", image_count)
print("Label text files:", label_count)
print("Empty label files:", empty_label_count)
print("YAML configuration files:", len(yaml_files))
print("Class IDs and annotation counts:", dict(sorted(class_counts.items())))
print("Potentially invalid label rows:", len(invalid_labels))

if invalid_labels:
    print("\nFirst 20 potential label issues:")
    for issue in invalid_labels[:20]:
        print(issue)

print("\nAudit complete. No files were modified.")