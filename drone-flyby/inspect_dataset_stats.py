from collections import defaultdict
from pathlib import Path
import csv
import json
import statistics


ANNOTATION_DIR = Path("src/helsinki/annotations")
DATASET_DIR = Path("datasets/helsinki")

IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT


def get_split(stem):
    """Determine whether this frame ended up in train or val."""
    for split in ("train", "val"):
        for ext in (".png", ".jpg", ".jpeg"):
            if (DATASET_DIR / "images" / split / f"{stem}{ext}").exists():
                return split
    return "unknown"


stats = defaultdict(lambda: {
    "instances": 0,
    "frames": set(),
    "train_instances": 0,
    "train_frames": set(),
    "val_instances": 0,
    "val_frames": set(),
    "unknown_instances": 0,
    "widths": [],
    "heights": [],
    "areas": [],
})


annotation_files = sorted(ANNOTATION_DIR.glob("frame_*.json"))

if not annotation_files:
    raise SystemExit(f"No annotations found in {ANNOTATION_DIR.resolve()}")


for annotation_path in annotation_files:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations = data.get("annotations", data.get("objects", []))

    frame = annotation_path.stem
    split = get_split(frame)

    for obj in annotations:
        cls = obj["object_id"]
        x1, y1, x2, y2 = map(float, obj["bbox"])

        width = x2 - x1
        height = y2 - y1
        area = width * height

        s = stats[cls]

        s["instances"] += 1
        s["frames"].add(frame)

        s["widths"].append(width)
        s["heights"].append(height)
        s["areas"].append(area)

        if split == "train":
            s["train_instances"] += 1
            s["train_frames"].add(frame)
        elif split == "val":
            s["val_instances"] += 1
            s["val_frames"].add(frame)
        else:
            s["unknown_instances"] += 1


rows = []

for cls, s in stats.items():
    areas_pct = [100 * a / IMAGE_AREA for a in s["areas"]]

    rows.append({
        "class": cls,
        "instances": s["instances"],
        "unique_frames": len(s["frames"]),
        "train_instances": s["train_instances"],
        "train_frames": len(s["train_frames"]),
        "val_instances": s["val_instances"],
        "val_frames": len(s["val_frames"]),
        "unknown_instances": s["unknown_instances"],
        "median_width_px": statistics.median(s["widths"]),
        "median_height_px": statistics.median(s["heights"]),
        "median_area_px2": statistics.median(s["areas"]),
        "median_area_pct": statistics.median(areas_pct),
        "min_area_px2": min(s["areas"]),
        "max_area_px2": max(s["areas"]),
    })


# Largest training set first
rows.sort(key=lambda x: (-x["instances"], x["class"]))


print()
print(
    f"{'CLASS':18} "
    f"{'INST':>5} "
    f"{'FRAMES':>6} "
    f"{'TRAIN':>6} "
    f"{'VAL':>5} "
    f"{'MED W':>7} "
    f"{'MED H':>7} "
    f"{'MED AREA':>10} "
    f"{'AREA %':>8}"
)

print("-" * 94)

for r in rows:
    print(
        f"{r['class']:18} "
        f"{r['instances']:5d} "
        f"{r['unique_frames']:6d} "
        f"{r['train_instances']:6d} "
        f"{r['val_instances']:5d} "
        f"{r['median_width_px']:7.1f} "
        f"{r['median_height_px']:7.1f} "
        f"{r['median_area_px2']:10.0f} "
        f"{r['median_area_pct']:7.3f}%"
    )


print("\nDetailed problem classes:")
print("=" * 80)

interesting = [
    "hangar",
    "medium_plane",
    "medium_launcher",
    "mine_roller",
    "small_launcher",
    "ta-ta",
]

by_class = {r["class"]: r for r in rows}

for cls in interesting:
    r = by_class.get(cls)

    if r is None:
        print(f"\n{cls}: NO ANNOTATIONS")
        continue

    print(f"\n{cls}")
    print(f"  instances:       {r['instances']}")
    print(f"  unique frames:   {r['unique_frames']}")
    print(f"  train instances: {r['train_instances']}")
    print(f"  train frames:    {r['train_frames']}")
    print(f"  val instances:   {r['val_instances']}")
    print(f"  val frames:      {r['val_frames']}")
    print(f"  median width:    {r['median_width_px']:.1f}px")
    print(f"  median height:   {r['median_height_px']:.1f}px")
    print(f"  median area:     {r['median_area_px2']:.0f}px²")
    print(f"  median frame %:  {r['median_area_pct']:.4f}%")
    print(f"  min area:        {r['min_area_px2']:.0f}px²")
    print(f"  max area:        {r['max_area_px2']:.0f}px²")


output = Path("dataset_class_stats.csv")

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved full table to: {output}")