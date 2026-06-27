"""
Bootstrap confidence intervals for kappa estimates.
Also computes balanced accuracy for elicitation vs ICL comparison.

Run locally:
  python bootstrap_ci.py --results_dir results
"""
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

def bootstrap_kappa(y_true, y_pred, n_bootstrap=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    boot_kappas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        if len(set(yt)) < 2:
            continue
        try:
            k = cohen_kappa_score(yt, yp)
            boot_kappas.append(k)
        except Exception:
            continue
    boot_kappas = np.array(boot_kappas)
    ci_low = np.percentile(boot_kappas, 2.5)
    ci_high = np.percentile(boot_kappas, 97.5)
    return float(np.mean(boot_kappas)), ci_low, ci_high

def load_icl_result(path):
    with open(path, 'r') as f:
        d = json.load(f)
    return d['all_true'], d['all_pred'], d['kappa']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--n_bootstrap', type=int, default=10000)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    LABEL2ID = {'de-escalating': 0, 'stable': 1, 'escalating': 2}
    ID2LABEL = {0: 'de-escalating', 1: 'stable', 2: 'escalating'}

    print("="*70)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%, n_bootstrap=10000)")
    print("="*70)

    # ICL results
    icl_files = {
        'Llama3 8B diff (corrected)': 'icl_diff_3shot_llama3_8b_results.json',
        'GPT-4 diff (corrected)':     'icl_diff_3shot_gpt_4_results.json',
        'GPT-4 raw':                  'icl_raw_3shot_gpt_4_results.json',
        'GPT-4o diff+KEY SIGNALS':    'icl_diff_3shot_gpt_4o_results.json',
    }

    icl_kappas = {}
    for label, fname in icl_files.items():
        fpath = results_dir / fname
        if not fpath.exists():
            print(f"  MISSING: {fname}")
            continue
        y_true, y_pred, kappa_point = load_icl_result(fpath)
        boot_mean, ci_low, ci_high = bootstrap_kappa(
            y_true, y_pred, args.n_bootstrap)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        icl_kappas[label] = {'kappa': kappa_point, 'ci_low': ci_low,
                              'ci_high': ci_high, 'bal_acc': bal_acc,
                              'y_true': y_true, 'y_pred': y_pred}
        print(f"\n{label}")
        print(f"  Kappa:             {kappa_point:.3f}")
        print(f"  95% CI:            [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  CI width:          {ci_high - ci_low:.3f}")
        print(f"  Balanced accuracy: {bal_acc:.3f}")

        # Per-class recall
        for cls_id, cls_name in ID2LABEL.items():
            yt = np.array(y_true)
            yp = np.array(y_pred)
            mask = yt == cls_id
            if mask.sum() > 0:
                recall = (yp[mask] == cls_id).sum() / mask.sum()
                print(f"  {cls_name:20s} recall: {recall:.3f} ({(yp[mask]==cls_id).sum()}/{mask.sum()})")

    # Elicitation result
    elicitation_path = results_dir / 'elicitation_gpt_4_results.json'
    print("\n" + "="*70)
    print("ELICITATION vs ICL — THRESHOLD-INDEPENDENT COMPARISON")
    print("="*70)

    if elicitation_path.exists():
        with open(elicitation_path, 'r') as f:
            elicit = json.load(f)

        # Convert elicitation YES/NO to 3-class for comparison
        # YES = de-escalating (0), NO = not de-escalating
        # For NO predictions: we don't know if model meant stable or escalating
        # So we compute binary balanced accuracy (de-esc vs not)
        records = elicit['records']

        # Binary: de-escalating vs not
        y_true_bin = [1 if r['human_label'] == 'de-escalating' else 0
                      for r in records]
        y_pred_bin = [1 if r['model_yes'] else 0 for r in records]

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_true_bin, y_pred_bin)
        except Exception:
            auc = None

        bal_acc_elicit = balanced_accuracy_score(y_true_bin, y_pred_bin)
        de_esc_recall = sum(1 for r in records
                           if r['human_label'] == 'de-escalating' and r['model_yes'])
        de_esc_total = sum(1 for r in records if r['human_label'] == 'de-escalating')
        non_de_esc_correct = sum(1 for r in records
                                  if r['human_label'] != 'de-escalating' and not r['model_yes'])
        non_de_esc_total = sum(1 for r in records if r['human_label'] != 'de-escalating')

        print(f"\nElicitation (GPT-4, YES/NO binary):")
        print(f"  De-esc recall (sensitivity):  {de_esc_recall}/{de_esc_total} ({de_esc_recall/de_esc_total:.3f})")
        print(f"  Non-de-esc recall (specificity): {non_de_esc_correct}/{non_de_esc_total} ({non_de_esc_correct/non_de_esc_total:.3f})")
        print(f"  Balanced accuracy:            {bal_acc_elicit:.3f}")
        if auc:
            print(f"  AUC:                          {auc:.3f}")

        # Compare to GPT-4 ICL on binary basis
        if 'GPT-4 diff (corrected)' in icl_kappas:
            icl_data = icl_kappas['GPT-4 diff (corrected)']
            yt = np.array(icl_data['y_true'])
            yp = np.array(icl_data['y_pred'])
            # Binary: de-esc (0) vs not
            yt_bin = (yt == 0).astype(int)
            yp_bin = (yp == 0).astype(int)
            bal_acc_icl = balanced_accuracy_score(yt_bin, yp_bin)
            icl_de_esc_recall = (yp_bin[yt_bin == 1] == 1).sum() / (yt_bin == 1).sum()
            icl_non_de_esc_recall = (yp_bin[yt_bin == 0] == 0).sum() / (yt_bin == 0).sum()
            print(f"\nGPT-4 diff ICL (corrected, binary de-esc vs not):")
            print(f"  De-esc recall (sensitivity):  {icl_de_esc_recall:.3f}")
            print(f"  Non-de-esc recall (specificity): {icl_non_de_esc_recall:.3f}")
            print(f"  Balanced accuracy:            {bal_acc_icl:.3f}")

            print(f"\nTHRESHOLD-INDEPENDENT COMPARISON:")
            print(f"  Elicitation balanced accuracy: {bal_acc_elicit:.3f}")
            print(f"  ICL diff balanced accuracy:    {bal_acc_icl:.3f}")
            delta = bal_acc_icl - bal_acc_elicit
            print(f"  Delta:                         {delta:+.3f}")
            if abs(delta) < 0.05:
                print(f"  INTERPRETATION: Essentially equivalent — the difference in")
                print(f"  de-esc recall between elicitation and ICL is a threshold shift,")
                print(f"  not a genuine improvement in discrimination ability.")
            elif delta > 0.05:
                print(f"  INTERPRETATION: ICL genuinely outperforms elicitation on")
                print(f"  balanced accuracy, not just a threshold shift.")
            else:
                print(f"  INTERPRETATION: Elicitation outperforms ICL on balanced accuracy.")
    else:
        print("  Elicitation results file not found.")

    # Overlap test: do Llama3 and GPT-4 CIs overlap?
    print("\n" + "="*70)
    print("CI OVERLAP: Llama3 8B vs GPT-4 (diff, corrected)")
    print("="*70)
    if ('Llama3 8B diff (corrected)' in icl_kappas and
            'GPT-4 diff (corrected)' in icl_kappas):
        l8b = icl_kappas['Llama3 8B diff (corrected)']
        g4 = icl_kappas['GPT-4 diff (corrected)']
        print(f"  Llama3 8B: {l8b['kappa']:.3f} [{l8b['ci_low']:.3f}, {l8b['ci_high']:.3f}]")
        print(f"  GPT-4:     {g4['kappa']:.3f} [{g4['ci_low']:.3f}, {g4['ci_high']:.3f}]")
        overlap = l8b['ci_low'] < g4['ci_high'] and g4['ci_low'] < l8b['ci_high']
        print(f"  CIs overlap: {overlap}")
        if overlap:
            print(f"  WARNING: Confidence intervals overlap — the kappa difference")
            print(f"  ({l8b['kappa'] - g4['kappa']:.3f}) is not statistically reliable at n=36.")
        else:
            print(f"  CIs do not overlap — difference is statistically reliable.")

if __name__ == '__main__':
    main()