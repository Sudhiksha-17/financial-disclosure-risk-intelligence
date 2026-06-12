import json

train = [json.loads(l) for l in open("outputs/finetune_train.jsonl")]
de = [ex for ex in train if ex["direction"] == "de-escalating"]
print(f"Total de-escalating: {len(de)}")
for ex in de:
    print(f"  {ex['pair_id']} | {ex['dim']}")