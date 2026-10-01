import json
import argparse
import itertools
from collections import defaultdict, Counter
from glob import glob

import numpy as np
from scipy import stats

import dataloader

DOMAINS = ("gender", "profession", "race", "religion")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="JSON do dataset (ex: dev_ptbr_v1.4.json).")
    p.add_argument("--predictions", default=None, help="Um arquivo de predições.")
    p.add_argument("--predictions-dir", default=None, help="Diretório com vários predictions_*.json.")
    p.add_argument("--output", default="chi2_domains.json")
    p.add_argument("--fdr-q", type=float, default=0.05, help="Nível de FDR, tanto para os 24 testes ômnibus quanto para as comparações par-a-par.")
    return p.parse_args()


def _example2sent(examples):
    m = {}
    for ex in examples:
        for s in ex.sentences:
            m[(ex.ID, s.gold_label)] = s.ID
    return m


def domain_counts(examples, id2score):
    e2s = _example2sent(examples)
    per_term = defaultdict(Counter)
    target2bias = {}

    for ex in examples:
        if ex.bias_type not in DOMAINS:
            continue
        key_pro, key_anti = (ex.ID, "stereotype"), (ex.ID, "anti-stereotype")
        if key_pro not in e2s or key_anti not in e2s:
            continue
        pro_id, anti_id = e2s[key_pro], e2s[key_anti]
        if pro_id not in id2score or anti_id not in id2score:
            continue
        pro, anti = id2score[pro_id], id2score[anti_id]
        per_term[ex.target]["pro" if pro > anti else "anti"] += 1
        per_term[ex.target]["total"] += 1
        target2bias[ex.target] = ex.bias_type

    counts = {d: Counter() for d in DOMAINS}
    for term, c in per_term.items():
        if c["total"] == 0:
            continue
        ss = c["pro"] / c["total"]
        domain = target2bias[term]
        if ss > 0.5:
            counts[domain]["pro"] += 1
        elif ss < 0.5:
            counts[domain]["anti"] += 1
    return counts


def load_id2score(predictions_path):
    with open(predictions_path, encoding="utf-8") as f:
        preds = json.load(f)
    id2score = {}
    for split in ("intrasentence", "intersentence"):
        for item in preds.get(split, []):
            id2score[item["id"]] = item["score"]
    return id2score


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


def chi2_omnibus(counts):
    table = [[counts[d]["pro"], counts[d]["anti"]] for d in DOMAINS]
    if any(sum(row) == 0 for row in table):
        return None
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "low_expected_cell_warning": bool((expected < 5).any()),
        "counts": {d: dict(counts[d]) for d in DOMAINS},
        "pct_estereotipada": {
            d: round(100 * counts[d]["pro"] / max(sum(counts[d].values()), 1), 2) for d in DOMAINS
        },
    }


def pairwise_fisher(counts, fdr_q):
    pairs = list(itertools.combinations(DOMAINS, 2))
    results, pvals = {}, []
    for a, b in pairs:
        table = [[counts[a]["pro"], counts[a]["anti"]], [counts[b]["pro"], counts[b]["anti"]]]
        if any(sum(row) == 0 for row in table):
            results[f"{a}_vs_{b}"] = None
            continue
        odds, p = stats.fisher_exact(table)
        results[f"{a}_vs_{b}"] = {"odds_ratio": float(odds), "p_value": float(p)}
        pvals.append(f"{a}_vs_{b}")

    if pvals:
        adj, rej = benjamini_hochberg([results[k]["p_value"] for k in pvals], fdr_q)
        for k, a_, r_ in zip(pvals, adj, rej):
            results[k]["p_adj_BH"] = round(a_, 4)
            results[k]["significant_BH"] = bool(r_)
    return results


def compute_omnibus_all(all_scores, examples_by_split, fdr_q):
    """
    Calcula o qui-quadrado ômnibus para todos os (modelo, split), aplica
    Benjamini-Hochberg sobre essa família completa de testes (mesmo espírito
    de stat_analysis.py para Wilcoxon/sinal), e só then decide para quais
    casos vale rodar o pós-hoc de Fisher.
    """
    flat = []  # (model_key, split_name, counts, omnibus)
    for model_key, id2score in all_scores.items():
        for split_name, examples in examples_by_split.items():
            counts = domain_counts(examples, id2score)
            omnibus = chi2_omnibus(counts)
            flat.append((model_key, split_name, counts, omnibus))

    pvals = [f[3]["p_value"] for f in flat if f[3] is not None]
    if pvals:
        adj, reject = benjamini_hochberg(pvals, fdr_q)
        it = iter(zip(adj, reject))
        for model_key, split_name, counts, omnibus in flat:
            if omnibus is None:
                continue
            a, r = next(it)
            omnibus["p_adj_BH"] = round(a, 4)
            omnibus["significant_BH"] = bool(r)

    out = defaultdict(dict)
    for model_key, split_name, counts, omnibus in flat:
        entry = {"omnibus_chi2": omnibus, "pairwise_fisher": None}
        if omnibus is not None and omnibus.get("significant_BH"):
            entry["pairwise_fisher"] = pairwise_fisher(counts, fdr_q)
        out[model_key][split_name] = entry
    return out


def main():
    args = parse_args()
    assert bool(args.predictions) != bool(args.predictions_dir), \
        "Forneça --predictions OU --predictions-dir, não ambos."

    dataset = dataloader.StereoSet(args.data)
    intra_ex = dataset.get_intrasentence_examples()
    inter_ex = dataset.get_intersentence_examples()
    examples_by_split = {"intrasentence": intra_ex, "intersentence": inter_ex,
                          "overall": intra_ex + inter_ex}

    files = glob(args.predictions_dir.rstrip("/") + "/*.json") if args.predictions_dir else [args.predictions]

    all_scores = {}
    for pf in files:
        model_key = pf.replace("predictions_", "").replace(".json", "").split("/")[-1].split("\\")[-1]
        all_scores[model_key] = load_id2score(pf)

    all_results = compute_omnibus_all(all_scores, examples_by_split, args.fdr_q)

    for model_key, res in all_results.items():
        print(f"\n=== {model_key} ===")
        for split_name in ("intrasentence", "intersentence", "overall"):
            r = res[split_name]
            om = r["omnibus_chi2"]
            if not om:
                print(f"  [{split_name}] dados insuficientes para o qui-quadrado.")
                continue
            sig = "significativo" if om["significant_BH"] else "NAO significativo"
            print(f"  [{split_name}] qui² entre domínios: chi2={om['chi2']:.2f}, dof={om['dof']}, "
                  f"p={om['p_value']:.4f}, p_FDR={om['p_adj_BH']:.4f} ({sig}, FDR sobre os 24 testes)")
            print(f"      % preferiu a estereotipada por domínio: {om['pct_estereotipada']}")
            if om["low_expected_cell_warning"]:
                print("      aviso: alguma célula esperada < 5 (comum em 'religion', que tem menos "
                      "exemplos); trate o p-valor com cautela.")
            if r["pairwise_fisher"]:
                print("      comparações par-a-par (Fisher exato, FDR):")
                for k, v in r["pairwise_fisher"].items():
                    if v is None:
                        continue
                    tag = "significativo" if v["significant_BH"] else "ns"
                    print(f"        {k}: OR={v['odds_ratio']:.2f}  p={v['p_value']:.4f}  "
                          f"p_FDR={v['p_adj_BH']:.4f} ({tag})")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
