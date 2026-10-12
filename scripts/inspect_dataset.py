from pathlib import Path
from collections import Counter
import json

root = Path(__file__).resolve().parents[1] / "data" / "dataset"
if not root.exists():
    raise SystemExit(f"Dataset folder not found: {root}")

exts = Counter()
examples = []
for p in root.rglob("*"):
    if p.is_file():
        exts[p.suffix.lower() or "[no extension]"] += 1
        if len(examples) < 100:
            examples.append(str(p.relative_to(root)))

print("Dataset root:", root)
print("File count:", sum(exts.values()))
print("Extensions:", json.dumps(exts, indent=2))
print("First paths:")
for p in examples:
    print(" ", p)
print("\nInspect the official README, licenses, split definitions, and mask encoding.")
print("Do not assume directory names or train on evaluation/test labels.")
