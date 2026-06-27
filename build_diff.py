"""
Diff Representation Builder v2
================================
Same as v1 but deduplicates on pair_id — takes the first occurrence
when the same pair appears in multiple input folders.

Also adds dimension-aware feature extraction:
- For credit/regulatory pairs: focuses on structural changes not COVID
- For operational pairs: includes COVID signal prominently
"""

import json
import re
import argparse
from pathlib import Path
from difflib import SequenceMatcher

STRONG_MODALS = [
    'will ', 'will not', 'has materially', 'is significantly',
    'continues to', 'is expected to continue', 'are currently',
    'is ongoing', 'remains', 'has had and will'
]

WEAK_MODALS = [
    'may ', 'could ', 'might ', 'has had and may', 'had and may',
    'may continue', 'could continue', 'previously', 'has previously',
    'historically', 'in the past'
]

COVID_TERMS = [
    'covid', 'pandemic', 'coronavirus', 'covid-19', 'variant',
    'vaccination', 'shelter', 'lockdown', 'quarantine', 'omicron',
    'delta variant', 'public health emergency'
]

def split_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z•\-])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def extract_headings(text):
    headings = set()
    for line in text.split('\n'):
        line = line.strip()
        if re.match(r'^[A-Z][A-Z\s\-,]{5,}$', line):
            headings.add(line[:100])
        elif re.match(r'Risks? (?:Related|Relating) to [^\n]+', line):
            headings.add(line[:100])
    return headings

def modal_strength(sentence):
    s = sentence.lower()
    strong = sum(1 for m in STRONG_MODALS if m in s)
    weak = sum(1 for m in WEAK_MODALS if m in s)
    if strong > weak: return 'strong'
    elif weak > strong: return 'weak'
    return 'neutral'

def is_covid_sentence(sentence):
    s = sentence.lower()
    return any(t in s for t in COVID_TERMS)

def sentence_similarity(s1, s2):
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def find_best_match(sentence, candidates, threshold=0.6):
    best_score, best_match = 0, None
    for c in candidates:
        score = sentence_similarity(sentence, c)
        if score > best_score:
            best_score, best_match = score, c
    if best_score >= threshold:
        return best_match, best_score
    return None, 0

def build_diff(earlier_text, later_text, dimension='operational_risk'):
    earlier_sents = split_sentences(earlier_text)
    later_sents = split_sentences(later_text)
    earlier_headings = extract_headings(earlier_text)
    later_headings = extract_headings(later_text)

    removed, matched_later = [], set()
    for s in earlier_sents:
        match, score = find_best_match(s, later_sents, threshold=0.65)
        if match is None:
            removed.append(s)
        else:
            idx = later_sents.index(match)
            matched_later.add(idx)

    added = []
    for i, s in enumerate(later_sents):
        if i not in matched_later:
            match, score = find_best_match(s, earlier_sents, threshold=0.65)
            if match is None:
                added.append(s)

    tense_shifts = []
    for s_early in earlier_sents:
        match, score = find_best_match(s_early, later_sents, threshold=0.65)
        if match and score < 0.95:
            early_strength = modal_strength(s_early)
            later_strength = modal_strength(match)
            if early_strength != later_strength:
                tense_shifts.append({
                    'earlier': s_early[:200],
                    'later': match[:200],
                    'change': f"{early_strength} -> {later_strength}"
                })

    added_headings = later_headings - earlier_headings
    removed_headings = earlier_headings - later_headings

    covid_removed = [s for s in removed if is_covid_sentence(s)]
    covid_added = [s for s in added if is_covid_sentence(s)]
    covid_tense_shifts = [t for t in tense_shifts if
                          is_covid_sentence(t['earlier']) or is_covid_sentence(t['later'])]

    # COVID signal
    covid_signal = 'none'
    if len(covid_removed) > len(covid_added):
        covid_signal = 'de-escalating'
    elif len(covid_added) > len(covid_removed):
        covid_signal = 'escalating'
    elif covid_tense_shifts:
        weakening = sum(1 for t in covid_tense_shifts if 'strong -> weak' in t['change'])
        strengthening = sum(1 for t in covid_tense_shifts if 'weak -> strong' in t['change'])
        if weakening > strengthening:
            covid_signal = 'de-escalating (tense only)'
        elif strengthening > weakening:
            covid_signal = 'escalating (tense only)'
        else:
            covid_signal = 'stable'

    # Structural change signal (for non-COVID pairs)
    structural_signal = 'stable'
    net_added = len(added) - len(removed)
    if net_added > 5:
        structural_signal = 'escalating'
    elif net_added < -5:
        structural_signal = 'de-escalating'
    if added_headings:
        structural_signal = 'escalating'
    if removed_headings and not added_headings:
        structural_signal = 'de-escalating'

    # Dimension-aware combined signal
    if dimension == 'operational_risk':
        primary_signal = covid_signal if covid_signal != 'none' else structural_signal
    else:
        # For credit/regulatory: structural changes matter more than COVID
        primary_signal = structural_signal if structural_signal != 'stable' else covid_signal

    # Build diff summary
    summary_parts = []
    if removed_headings:
        summary_parts.append(f"REMOVED SECTIONS: {'; '.join(list(removed_headings)[:3])}")
    if added_headings:
        summary_parts.append(f"ADDED SECTIONS: {'; '.join(list(added_headings)[:3])}")
    if covid_removed:
        summary_parts.append(f"COVID LANGUAGE REMOVED ({len(covid_removed)} sentences): " +
                            covid_removed[0][:150])
    if covid_added:
        summary_parts.append(f"COVID LANGUAGE ADDED ({len(covid_added)} sentences): " +
                            covid_added[0][:150])
    if covid_tense_shifts:
        t = covid_tense_shifts[0]
        summary_parts.append(f"COVID TENSE SHIFT ({t['change']}): "
                            f"'{t['earlier'][:100]}' -> '{t['later'][:100]}'")
    # Filter boilerplate artifacts before including in summary
    boilerplate_patterns = [
        r'^see the table', r'^see note', r'^table of contents',
        r'^\d+\s+table of', r'^for reconciliation', r'^page \d+',
        r'^in addition, our operating results for',
        r'^our operating results for \d{4}',
    ]
    def is_boilerplate(s):
        sl = s.lower().strip()
        return any(re.match(p, sl) for p in boilerplate_patterns) or len(sl) < 30

    non_covid_removed = [s for s in removed
                         if not is_covid_sentence(s) and not is_boilerplate(s)][:3]
    if non_covid_removed:
        summary_parts.append(f"OTHER REMOVED: " + " | ".join(s[:100] for s in non_covid_removed))
    non_covid_added = [s for s in added
                       if not is_covid_sentence(s) and not is_boilerplate(s)][:3]
    if non_covid_added:
        summary_parts.append(f"OTHER ADDED: " + " | ".join(s[:100] for s in non_covid_added))

    diff_summary = "\n".join(summary_parts) if summary_parts else "No significant structural changes detected."

    return {
        'added_sentences': added,
        'removed_sentences': removed,
        'tense_shifts': tense_shifts,
        'added_headings': list(added_headings),
        'removed_headings': list(removed_headings),
        'covid_removed': covid_removed,
        'covid_added': covid_added,
        'covid_tense_shifts': covid_tense_shifts,
        'covid_signal': covid_signal,
        'structural_signal': structural_signal,
        'primary_signal': primary_signal,
        'diff_summary': diff_summary,
        'stats': {
            'earlier_sentences': len(earlier_sents),
            'later_sentences': len(later_sents),
            'added_count': len(added),
            'removed_count': len(removed),
            'tense_shift_count': len(tense_shifts),
            'covid_removed_count': len(covid_removed),
            'covid_added_count': len(covid_added),
        }
    }

def parse_annotations(annotations_path):
    annotations = {}
    content = Path(annotations_path).read_text(encoding='utf-8')
    blocks = re.split(r'\n-{48}\n', content)
    for block in blocks:
        pair_match = re.search(r'PAIR:\s+(\S+)', block)
        dim_match = re.search(r'DIMENSION:\s+(\S+)', block)
        dir_match = re.search(r'direction\s*:\s*(\S+)', block)
        int_match = re.search(r'intensity\s*:\s*(\S+)', block)
        reason_match = re.search(r'reason\s*:(.*?)(?=\n\d+\.|={48}|$)', block, re.DOTALL)
        if pair_match and dir_match:
            pair_id = pair_match.group(1)
            direction = dir_match.group(1).lower()
            if direction in ['de-escalating', 'escalating', 'stable']:
                if pair_id not in annotations:  # First occurrence wins
                    annotations[pair_id] = {
                        'pair_id': pair_id,
                        'dimension': dim_match.group(1) if dim_match else '',
                        'direction': direction,
                        'intensity': int_match.group(1) if int_match else '',
                        'reason': reason_match.group(1).strip() if reason_match else ''
                    }
    return annotations

def parse_extracted_pair(filepath):
    content = Path(filepath).read_text(encoding='utf-8')
    earlier_match = re.search(
        r'={70}\nEARLIER \(\d{4}\)\n={70}\n(.*?)\n={70}',
        content, re.DOTALL)
    later_match = re.search(
        r'={70}\nLATER \(\d{4}\)\n={70}\n(.*?)\n={70}',
        content, re.DOTALL)
    if not earlier_match or not later_match:
        return None, None
    return earlier_match.group(1).strip(), later_match.group(1).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+',
                        default=['deescalating_v7','deescalating_v8',
                                 'deescalating_v9','extracted_missing'])
    parser.add_argument('--annotations', default='annotations_final_v3.txt')
    parser.add_argument('--output_dir', default='diffs')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Loading annotations from {args.annotations}...")
    annotations = parse_annotations(args.annotations)
    print(f"  Found {len(annotations)} unique usable annotations")

    results = []
    processed_pairs = set()  # DEDUPLICATION KEY

    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  WARNING: {input_dir} not found, skipping")
            continue

        txt_files = list(input_path.glob('*.txt'))
        print(f"\nProcessing {len(txt_files)} files from {input_dir}...")

        for fpath in sorted(txt_files):
            stem = fpath.stem
            parts = stem.split('_')
            try:
                pair_id = '_'.join(parts[:4])
            except:
                continue

            if pair_id not in annotations:
                continue

            # DEDUPLICATE - skip if already processed
            if pair_id in processed_pairs:
                print(f"  Skipping duplicate: {pair_id}")
                continue
            processed_pairs.add(pair_id)

            ann = annotations[pair_id]
            earlier_text, later_text = parse_extracted_pair(fpath)
            if not earlier_text or not later_text:
                continue

            print(f"  Building diff for {pair_id}...")
            diff = build_diff(earlier_text, later_text, ann['dimension'])

            record = {**ann, 'diff': diff, 'source_file': str(fpath)}
            results.append(record)
            print(f"    {ann['direction']} | "
                  f"added={diff['stats']['added_count']} "
                  f"removed={diff['stats']['removed_count']} "
                  f"covid={diff['covid_signal']} "
                  f"structural={diff['structural_signal']} "
                  f"primary={diff['primary_signal']}")

    # Save JSONL
    jsonl_path = out_dir / 'diff_representations.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\nSaved {len(results)} unique diff representations to {jsonl_path}")

    # Save readable
    readable_path = out_dir / 'diff_representations_readable.txt'
    with open(readable_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"{'='*70}\n")
            f.write(f"PAIR:      {r['pair_id']}\n")
            f.write(f"DIMENSION: {r['dimension']}\n")
            f.write(f"LABEL:     {r['direction']} (intensity={r['intensity']})\n")
            f.write(f"COVID SIG: {r['diff']['covid_signal']}\n")
            f.write(f"STRUCT SIG:{r['diff']['structural_signal']}\n")
            f.write(f"PRIMARY:   {r['diff']['primary_signal']}\n")
            f.write(f"STATS:     added={r['diff']['stats']['added_count']} "
                    f"removed={r['diff']['stats']['removed_count']} "
                    f"tense_shifts={r['diff']['stats']['tense_shift_count']}\n")
            f.write(f"\nDIFF SUMMARY:\n{r['diff']['diff_summary']}\n")
            f.write(f"\nHUMAN REASON:\n{r['reason']}\n\n")

    print(f"Saved readable to {readable_path}")

    from collections import Counter
    dist = Counter(r['direction'] for r in results)
    print(f"\nClass distribution: {dict(dist)}")

    # Alignment check - primary signal
    def aligned(label, signal):
        if label == 'de-escalating' and ('de-escalating' in signal or signal == 'none'):
            return True
        if label == 'escalating' and 'escalating' in signal:
            return True
        if label == 'stable' and signal in ('none', 'stable'):
            return True
        return False

    covid_align = sum(1 for r in results
                      if aligned(r['direction'], r['diff']['covid_signal']))
    primary_align = sum(1 for r in results
                        if aligned(r['direction'], r['diff']['primary_signal']))
    print(f"COVID signal alignment:   {covid_align}/{len(results)} ({100*covid_align/len(results):.1f}%)")
    print(f"Primary signal alignment: {primary_align}/{len(results)} ({100*primary_align/len(results):.1f}%)")

    print(f"\nMisalignments (primary signal):")
    for r in results:
        if not aligned(r['direction'], r['diff']['primary_signal']):
            print(f"  {r['pair_id']}: human={r['direction']}, primary={r['diff']['primary_signal']}")

if __name__ == '__main__':
    main()