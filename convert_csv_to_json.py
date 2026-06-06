"""
Converte o CSV traduzido para o formato JSON do StereoSet.

Uso:
    python convert_csv_to_json.py --input ptbr_llm.csv --output data/dev_ptbr.json
"""

import csv
import json
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../ptbr_llm.csv")
    parser.add_argument("--output", default="data/dev_ptbr.json")
    return parser.parse_args()


def convert(input_path: str, output_path: str):
    clusters: dict[str, dict] = {}

    with open(input_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = row["cluster_id"]
            dtype = row["dataset_type"]

            if cid not in clusters:
                clusters[cid] = {
                    "id": cid,
                    "target": row["target"],
                    "bias_type": row["bias_type"],
                    "context": row["context_pt"],
                    "sentences": [],
                    "_type": dtype,
                }

            clusters[cid]["sentences"].append({
                "sentence": row["sentence_pt"],
                "gold_label": row["gold_label"],
                "id": row["sentence_id"],
            })

    intrasentence = []
    intersentence = []

    for cluster in clusters.values():
        dtype = cluster.pop("_type")
        entry = {
            "id": cluster["id"],
            "target": cluster["target"],
            "bias_type": cluster["bias_type"],
            "context": cluster["context"],
            "sentences": cluster["sentences"],
        }
        if dtype == "intrasentence":
            intrasentence.append(entry)
        else:
            intersentence.append(entry)

    output = {
        "version": "1.0-ptbr",
        "data": {
            "intrasentence": intrasentence,
            "intersentence": intersentence,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Convertido: {len(intrasentence)} intrasentence, {len(intersentence)} intersentence")
    print(f"Salvo em: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)
