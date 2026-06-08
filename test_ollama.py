import json
from pathlib import Path
from collections import Counter

signals_dir = Path("data/processed/risk_signals")
files = list(signals_dir.glob("*.json"))
print(f"Total signal files: {len(files)}")

directions  = Counter()
intensities = []

for f in files:
    data = json.loads(f.read_text(encoding="utf-8"))
    for dim, vals in data["signals"].items():
        directions[f"{dim}:{vals['direction']}"] += 1
        intensities.append(vals["intensity"])

print(f"Average intensity: {sum(intensities)/len(intensities):.2f}")
print(f"High intensity (4-5): {sum(1 for i in intensities if i >= 4)}")
print()
print("Top 10 direction counts:")
for k, v in directions.most_common(10):
    print(f"  {k}: {v}")