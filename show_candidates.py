import json
from pathlib import Path

pairs_dir   = Path("data/processed/pairs")
output_dir  = Path("annotation_candidates")
output_dir.mkdir(exist_ok=True)

targets = [
    ("VLO_10-K_2021_2022",  "credit_risk"),
    ("CINF_10-K_2020_2021", "regulatory_risk"),
    ("CINF_10-K_2021_2022", "regulatory_risk"),
    ("FITB_10-K_2022_2023", "regulatory_risk"),
    ("MPC_10-K_2019_2020",  "credit_risk"),
    ("MPC_10-K_2020_2021",  "credit_risk"),
    ("MPC_10-K_2022_2023",  "regulatory_risk"),
    ("TWLO_10-K_2019_2020", "credit_risk"),
    ("ZS_10-K_2020_2021",   "credit_risk"),
    ("FANG_10-K_2020_2021", "operational_risk"),
    ("FANG_10-K_2021_2022", "operational_risk"),
    ("FANG_10-K_2022_2023", "operational_risk"),
    ("HUBS_10-K_2021_2022", "operational_risk"),
    ("AIZ_10-K_2019_2020",  "operational_risk"),
    ("CINF_10-K_2022_2023", "credit_risk"),
    ("EOG_10-K_2019_2020",  "credit_risk"),
    ("AIG_10-K_2019_2020",  "credit_risk"),
    ("AIG_10-K_2020_2021",  "credit_risk"),
    ("AIG_10-K_2021_2022",  "credit_risk"),
    ("AIG_10-K_2022_2023",  "credit_risk"),
    ("PRU_10-K_2019_2020",  "operational_risk"),
    ("PRU_10-K_2020_2021",  "operational_risk"),
]

for i, (pair_id, dim) in enumerate(targets, 1):
    pair_file = pairs_dir / f"{pair_id}.json"
    if not pair_file.exists():
        print(f"[{i}] MISSING: {pair_id}")
        continue

    pair   = json.loads(pair_file.read_text(encoding="utf-8"))
    ticker = pair.get("ticker")
    ye     = pair.get("year_earlier")
    yl     = pair.get("year_later")
    te     = pair.get("earlier", {}).get("text", "")
    tl     = pair.get("later",   {}).get("text", "")

    content = f"""PAIR:      {pair_id}
DIMENSION: {dim}
TICKER:    {ticker} | {ye} -> {yl}

{'='*70}
EARLIER ({ye})
{'='*70}
{te}

{'='*70}
LATER ({yl})
{'='*70}
{tl}

{'='*70}
ANNOTATE
{'='*70}
  direction  : 
  intensity  : 
  confident  : 
  reason     : 
"""

    out_file = output_dir / f"{i:02d}_{pair_id}_{dim}.txt"
    out_file.write_text(content, encoding="utf-8")
    print(f"[{i}] Written: {out_file.name}")

print(f"\nDone. Open files in: {output_dir.resolve()}")