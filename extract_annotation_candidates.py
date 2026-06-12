import json
from pathlib import Path

pairs_dir = Path("data/processed/pairs")

targets = [
    ("HBAN_10-K_2021_2022", "credit_risk"),
    ("EOG_10-K_2020_2021",  "market_risk"),
    ("OKTA_10-K_2020_2021", "credit_risk"),
    ("VLO_10-K_2021_2022",  "credit_risk"),
    ("VLO_10-K_2022_2023",  "credit_risk"),
    ("CINF_10-K_2020_2021", "regulatory_risk"),
    ("CINF_10-K_2021_2022", "regulatory_risk"),
    ("TWLO_10-K_2019_2020", "credit_risk"),
    ("ZS_10-K_2020_2021",   "credit_risk"),
    ("THG_10-K_2022_2023",  "credit_risk"),
    ("FITB_10-K_2022_2023", "regulatory_risk"),
    ("MPC_10-K_2019_2020",  "credit_risk"),
    ("MPC_10-K_2022_2023",  "regulatory_risk"),
    ("FANG_10-K_2020_2021", "operational_risk"),
    ("HUBS_10-K_2021_2022", "operational_risk"),
    ("PRU_10-K_2023_2024",  "operational_risk"),
    ("AIZ_10-K_2019_2020",  "operational_risk"),
    ("NTRS_10-K_2019_2020", "operational_risk"),
    ("HBAN_10-K_2023_2024", "regulatory_risk"),
    ("ZION_10-K_2019_2020", "operational_risk"),
]

print(f"Extracting {len(targets)} annotation candidates\n")
print(f"{'Pair ID':35s} {'Dimension':20s} {'Earlier':6s} {'Later':6s}")
print("-" * 80)

for pair_id, dim in targets:
    pair_file = pairs_dir / f"{pair_id}.json"
    if not pair_file.exists():
        print(f"  MISSING: {pair_id}")
        continue

    pair = json.loads(pair_file.read_text(encoding="utf-8"))
    ye   = pair.get("year_earlier")
    yl   = pair.get("year_later")
    te   = pair.get("earlier", {}).get("text", "")[:800]
    tl   = pair.get("later",   {}).get("text", "")[:800]

    print(f"\n{'='*80}")
    print(f"PAIR:      {pair_id}")
    print(f"DIMENSION: {dim}")
    print(f"TICKER:    {pair.get('ticker')} | {ye} -> {yl}")
    print(f"\nEARLIER TEXT (first 800 chars):")
    print(te)
    print(f"\nLATER TEXT (first 800 chars):")
    print(tl)
    print(f"\nYOUR ANNOTATION:")
    print(f"  direction  : escalating / stable / de-escalating")
    print(f"  intensity  : 1-5")
    print(f"  confident? : yes / no")