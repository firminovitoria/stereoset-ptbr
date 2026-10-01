import json
import argparse
from collections import defaultdict, Counter
from glob import glob

import numpy as np
from scipy import stats

import dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="JSON do dataset (ex: dev_ptbr_v1.4.json).")
    p.add_argument("--predictions", default=None, help="Um arquivo de predições.")
    p.add_argument("--predictions-dir", default=None, help="Diretório com vários predictions_*.json.")
    p.add_argument("--output", default="stats.json")
    p.add_argument("--bootstrap", type=int, default=2000, help="Nº de reamostragens do bootstrap.")
    p.add_argument("--fdr-q", type=float, default=0.05, help="Nível de FDR (Benjamini-Hochberg) para os testes vs. chance.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", nargs=2, default=None, metavar=("PRED_A", "PRED_B"),
                    help="Compara dois arquivos de predições (mesmo --data) com Wilcoxon pareado.")
    p.add_argument("--compare-data", default=None, help="Segunda versão do dataset, para comparar antes/depois.")
    p.add_argument("--compare-predictions", default=None, help="Predições correspondentes ao --compare-data.")
    return p.parse_args()


def _example2sent(examples):
    m = {}
    for ex in examples:
        for s in ex.sentences:
            m[(ex.ID, s.gold_label)] = s.ID
    return m


def per_term_scores(examples, id2score):
    e2s = _example2sent(examples)
    per_term = defaultdict(Counter)
    target2bias = {}

    for ex in examples:
        key_pro, key_anti, key_unr = (ex.ID, "stereotype"), (ex.ID, "anti-stereotype"), (ex.ID, "unrelated")
        if not all(k in e2s for k in (key_pro, key_anti, key_unr)):
            continue
        pro_id, anti_id, unr_id = e2s[key_pro], e2s[key_anti], e2s[key_unr]
        if not all(i in id2score for i in (pro_id, anti_id, unr_id)):
            continue

        pro, anti, unr = id2score[pro_id], id2score[anti_id], id2score[unr_id]
        per_term[ex.target]["pro" if pro > anti else "anti"] += 1
        if pro > unr:
            per_term[ex.target]["related"] += 1
        if anti > unr:
            per_term[ex.target]["related"] += 1
        per_term[ex.target]["total"] += 1
        target2bias[ex.target] = ex.bias_type

    ss_by_term, lm_by_term = {}, {}
    for term, c in per_term.items():
        if c["total"] == 0:
            continue
        ss_by_term[term] = 100.0 * c["pro"] / c["total"]
        lm_by_term[term] = 100.0 * c["related"] / (c["total"] * 2)

    return ss_by_term, lm_by_term, target2bias


def load_id2score(predictions_path):
    with open(predictions_path, encoding="utf-8") as f:
        preds = json.load(f)
    id2score = {}
    for split in ("intrasentence", "intersentence"):
        for item in preds.get(split, []):
            id2score[item["id"]] = item["score"]
    return id2score


def bootstrap_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float(values[0]), float(values[0])) if len(values) else (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(values, n_boot, seed):
    values = list(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    lo, hi = bootstrap_ci(values, n_boot, seed)
    return {"mean": round(mean, 2), "std": round(std, 2), "n_terms": len(values),
            "CI95": [round(lo, 2), round(hi, 2)]}


def wilcoxon_vs_chance(ss_values):
    diffs = np.asarray(ss_values, dtype=float) - 50.0
    diffs = diffs[diffs != 0]
    if len(diffs) < 5:
        return None
    stat, p = stats.wilcoxon(diffs)
    return {"statistic": float(stat), "p_value": float(p), "n": len(diffs)}


def sign_test_vs_chance(ss_values):
    diffs = np.asarray(ss_values, dtype=float) - 50.0
    diffs = diffs[diffs != 0]
    if len(diffs) < 5:
        return None
    n_pos = int((diffs > 0).sum())
    n = len(diffs)
    res = stats.binomtest(n_pos, n, 0.5, alternative="two-sided")
    return {"n_pos": n_pos, "n_neg": n - n_pos, "n": n, "p_value": float(res.pvalue)}


def benjamini_hochberg(pvals, q=0.05):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adj_ranked = ranked * m / (np.arange(m) + 1)
    adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]
    adj_ranked = np.clip(adj_ranked, 0, 1)

    adj = np.empty(m)
    adj[order] = adj_ranked

    reject = adj <= q
    return adj.tolist(), reject.tolist()


def analyze_model(examples_by_split, id2score, n_boot, seed):
    out = {}
    for split_name, examples in examples_by_split.items():
        ss_t, lm_t, _ = per_term_scores(examples, id2score)
        if not ss_t:
            out[split_name] = None
            continue
        ss_vals, lm_vals = list(ss_t.values()), list(lm_t.values())
        out[split_name] = {
            "SS": summarize(ss_vals, n_boot, seed),
            "LM": summarize(lm_vals, n_boot, seed),
            "SS_vs_chance_wilcoxon": wilcoxon_vs_chance(ss_vals),
            "SS_vs_chance_sign_test": sign_test_vs_chance(ss_vals),
            "_ss_by_term": ss_t,
        }
    return out


def apply_fdr_correction(all_results, q, test_key):
    flat = []
    for model_key, res in all_results.items():
        if model_key.startswith("_"):
            continue
        for split_name in ("intrasentence", "intersentence", "overall"):
            r = res.get(split_name)
            if not r or not r.get(test_key):
                continue
            flat.append((model_key, split_name, r[test_key]["p_value"]))

    if len(flat) < 2:
        return

    pvals = [p for _, _, p in flat]
    adj, reject = benjamini_hochberg(pvals, q)

    for (model_key, split_name, _), a, rej in zip(flat, adj, reject):
        all_results[model_key][split_name][test_key]["p_adj_BH"] = round(a, 4)
        all_results[model_key][split_name][test_key]["significant_BH"] = bool(rej)

    print(f"\nCorreção FDR (Benjamini-Hochberg, q={q}) aplicada a {len(flat)} testes de '{test_key}'.")


def paired_wilcoxon(ss_by_term_a, ss_by_term_b, label_a="A", label_b="B"):
    common = sorted(set(ss_by_term_a) & set(ss_by_term_b))
    if len(common) < 5:
        print(f"  Poucos termos em comum ({len(common)}) entre {label_a} e {label_b}; teste não é confiável.")
        return None
    a = np.array([ss_by_term_a[t] for t in common])
    b = np.array([ss_by_term_b[t] for t in common])
    diff = a - b
    if np.all(diff == 0):
        return {"n_terms": len(common), "statistic": 0.0, "p_value": 1.0,
                "mean_diff_SS": 0.0}
    stat, p = stats.wilcoxon(diff)
    return {"n_terms": len(common), "statistic": float(stat), "p_value": float(p),
            "mean_diff_SS": round(float(diff.mean()), 2)}


def main():
    args = parse_args()
    assert bool(args.predictions) != bool(args.predictions_dir) or args.compare_data, \
        "Forneça --predictions OU --predictions-dir (a menos que use --compare-data)."

    dataset = dataloader.StereoSet(args.data)
    intra_ex = dataset.get_intrasentence_examples()
    inter_ex = dataset.get_intersentence_examples()
    examples_by_split = {"intrasentence": intra_ex, "intersentence": inter_ex,
                          "overall": intra_ex + inter_ex}

    all_results = {}
    all_ss_by_term = {}

    if args.compare:
        files = list(args.compare)
    else:
        files = glob(args.predictions_dir.rstrip("/") + "/*.json") if args.predictions_dir else [args.predictions]

    for pf in files:
        model_key = pf.replace("predictions_", "").replace(".json", "").split("/")[-1].split("\\")[-1]
        print(f"\n=== {model_key} ===")
        id2score = load_id2score(pf)
        res = analyze_model(examples_by_split, id2score, args.bootstrap, args.seed)
        all_ss_by_term[model_key] = res["overall"]["_ss_by_term"] if res["overall"] else {}
        all_results[model_key] = res

    apply_fdr_correction(all_results, args.fdr_q, "SS_vs_chance_wilcoxon")
    apply_fdr_correction(all_results, args.fdr_q, "SS_vs_chance_sign_test")

    for model_key, res in all_results.items():
        print(f"\n=== {model_key} ===")
        for split_name in ("intrasentence", "intersentence", "overall"):
            r = res[split_name]
            if not r:
                continue
            print(f"  [{split_name}] SS = {r['SS']['mean']} ± {r['SS']['std']}  "
                  f"(IC95% {r['SS']['CI95']}, n={r['SS']['n_terms']} termos)   "
                  f"LM = {r['LM']['mean']} ± {r['LM']['std']}")
            wc = r["SS_vs_chance_wilcoxon"]
            if wc:
                sig = "significativo" if wc["p_value"] < 0.05 else "NAO significativo"
                sig_fdr = "significativo" if wc["significant_BH"] else "NAO significativo"
                print(f"      Wilcoxon vs. chance (50): p = {wc['p_value']:.4f} ({sig}, n={wc['n']})  |  "
                      f"p_FDR = {wc['p_adj_BH']:.4f} ({sig_fdr})")
            sg = r["SS_vs_chance_sign_test"]
            if sg:
                sig = "significativo" if sg["p_value"] < 0.05 else "NAO significativo"
                sig_fdr = "significativo" if sg["significant_BH"] else "NAO significativo"
                print(f"      Sign test vs. chance    : p = {sg['p_value']:.4f} ({sig}, {sg['n_pos']}+/{sg['n_neg']}-)  |  "
                      f"p_FDR = {sg['p_adj_BH']:.4f} ({sig_fdr})")
            r.pop("_ss_by_term")

    if args.compare and len(all_ss_by_term) == 2:
        (ka, sa), (kb, sb) = list(all_ss_by_term.items())
        print(f"\n=== Comparação pareada (overall): {ka} vs {kb} ===")
        cmp = paired_wilcoxon(sa, sb, ka, kb)
        if cmp:
            sig = "significativo" if cmp["p_value"] < 0.05 else "NAO significativo"
            print(f"  Wilcoxon pareado: p = {cmp['p_value']:.4f}  ({sig}, n={cmp['n_terms']} termos em comum)")
            print(f"  Diferença média de SS ({ka} - {kb}): {cmp['mean_diff_SS']}")
        all_results["_comparacao_pareada"] = {f"{ka}_vs_{kb}": cmp}

    if args.compare_data and args.compare_predictions:
        print(f"\n=== Comparação antes/depois da correção (mesmo modelo) ===")
        dataset2 = dataloader.StereoSet(args.compare_data)
        ex2 = dataset2.get_intrasentence_examples() + dataset2.get_intersentence_examples()
        id2score2 = load_id2score(args.compare_predictions)
        ss_before, _, _ = per_term_scores(ex2, id2score2)
        model_key = list(all_ss_by_term.keys())[0]
        ss_after = all_ss_by_term[model_key]
        cmp = paired_wilcoxon(ss_after, ss_before, "depois", "antes")
        if cmp:
            sig = "significativo" if cmp["p_value"] < 0.05 else "NAO significativo"
            print(f"  Wilcoxon pareado (depois vs antes): p = {cmp['p_value']:.4f}  ({sig}, n={cmp['n_terms']})")
            print(f"  Diferença média de SS (depois - antes): {cmp['mean_diff_SS']}")
        all_results["_comparacao_antes_depois"] = cmp

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
