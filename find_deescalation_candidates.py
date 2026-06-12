import json
from pathlib import Path

signals_dir = Path("data/processed/risk_signals")
pairs_dir   = Path("data/processed/pairs")

candidates = []

for sig_file in sorted(signals_dir.glob("*.json")):
    sig  = json.loads(sig_file.read_text(encoding="utf-8"))
    pair_id      = sig["pair_id"]
    year_earlier = sig.get("year_earlier", 0)
    year_later   = sig.get("year_later", 0)

    # Focus on 2020-2023 transitions
    if not (year_earlier in [2019, 2020, 2021, 2022] and
            year_later   in [2020, 2021, 2022, 2023, 2024]):
        continue

    pair_file = pairs_dir / f"{pair_id}.json"
    if not pair_file.exists():
        continue

    pair = json.loads(pair_file.read_text(encoding="utf-8"))
    earlier_text = pair.get("earlier", {}).get("text", "").lower()
    later_text   = pair.get("later",   {}).get("text", "").lower()

    if not earlier_text or not later_text:
        continue

    # Count risk-related terms in each filing
    risk_terms = [
        "covid", "pandemic", "lockdown", "quarantine",
        "libor", "cessation", "transition",
        "merger", "acquisition pending",
        "regulatory approval"
    ]

    earlier_count = sum(earlier_text.count(t) for t in risk_terms)
    later_count   = sum(later_text.count(t) for t in risk_terms)

    # Significant reduction in risk language
    if earlier_count >= 3 and later_count <= 1:
        candidates.append({
            "pair_id":      pair_id,
            "ticker":       sig.get("ticker"),
            "year_earlier": year_earlier,
            "year_later":   year_later,
            "earlier_count": earlier_count,
            "later_count":   later_count,
            "ratio":         round(later_count / max(earlier_count, 1), 2)
        })

# Sort by biggest reduction
candidates.sort(key=lambda x: x["earlier_count"], reverse=True)

print(f"De-escalation candidates: {len(candidates)}")
for c in candidates[:40]:
    print(f"  {c['ticker']:8s} {c['year_earlier']}->{c['year_later']} | "
          f"terms: {c['earlier_count']} -> {c['later_count']}")