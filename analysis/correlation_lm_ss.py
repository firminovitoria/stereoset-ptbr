import json
import argparse

import numpy as np
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_v1.4.json", help="Saída da evaluation.py.")
    p.add_argument("--output", default="correlation_lm_ss.json")
    return p.parse_args()


def collect_points(results, split):
    points = []
    for model_key, metrics in results.items():
        if split == "overall":
            m = metrics.get("overall")
        else:
            split_dict = metrics.get(split)
            m = split_dict.get("overall") if split_dict else None
        if not m or "LM Score" not in m or "SS Score" not in m:
            continue
        points.append((model_key, m["LM Score"], m["SS Score"]))
    return points


def correlate(points):
    if len(points) < 4:
        return None
    lm = np.array([p[1] for p in points], dtype=float)
    ss = np.array([p[2] for p in points], dtype=float)

    rho, p_spearman = stats.spearmanr(lm, ss)
    r, p_pearson = stats.pearsonr(lm, ss)

    return {
        "n_modelos": len(points),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p_spearman), 4),
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p_pearson), 4),
        "pontos": {p[0]: {"LM": p[1], "SS": p[2]} for p in points},
    }


def main():
    args = parse_args()
    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)
    results = {k: v for k, v in results.items() if not k.startswith("_")}

    out = {}
    for split in ("intrasentence", "intersentence", "overall"):
        points = collect_points(results, split)
        res = correlate(points)
        out[split] = res
        if not res:
            print(f"\n[{split}] dados insuficientes ({len(points)} modelos).")
            continue

        sig = "significativo" if res["spearman_p"] < 0.05 else "NAO significativo (cuidado: n pequeno)"
        print(f"\n[{split}] n={res['n_modelos']} modelos")
        print(f"  Spearman: rho = {res['spearman_rho']}  p = {res['spearman_p']}  ({sig})")
        print(f"  Pearson : r   = {res['pearson_r']}  p = {res['pearson_p']}")
        ordered = sorted(res["pontos"].items(), key=lambda kv: kv[1]["LM"])
        for model, v in ordered:
            print(f"    {model:20s} LM={v['LM']:6.2f}  SS={v['SS']:6.2f}")

    print(
        "\nAviso: com n=8 modelos, este teste so detecta correlacoes fortes; "
        "trate o p-valor como indicativo e reporte o rho + o grafico de dispersao no artigo, "
        "nao so o p-valor isolado."
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
