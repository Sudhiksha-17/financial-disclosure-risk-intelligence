import json
from pathlib import Path

signals_dir = Path("data/processed/risk_signals")

candidates = []

for sig_file in sorted(signals_dir.glob("*.json")):
    sig     = json.loads(sig_file.read_text(encoding="utf-8"))
    pair_id = sig["pair_id"]
    ticker  = sig.get("ticker")
    ye      = sig.get("year_earlier")
    yl      = sig.get("year_later")
    signals = sig.get("signals", {})

    for dim, vals in signals.items():
        if vals.get("direction") == "de-escalating":
            candidates.append({
                "pair_id": pair_id,
                "ticker":  ticker,
                "ye":      ye,
                "yl":      yl,
                "dim":     dim,
                "just":    vals.get("justification", "")
            })

print(f"LLM predicted de-escalating: {len(candidates)} cases")
print()
for c in candidates:
    print(f"  {c['ticker']:8s} {c['ye']}->{c['yl']} | {c['dim']:20s} | {c['just']}")