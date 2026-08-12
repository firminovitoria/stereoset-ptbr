"""
Reporta a distribuição do número de subtokens que o termo-alvo (a palavra que
preenche o BLANK) ocupa em cada tokenizador, na tarefa intra-sentence.

Endereça o Comentário do professor (revisão de qualificação, Seção 3.1.3):
"eu não encontrei no texto quantos subtokens o termo-alvo ocupa". Usa apenas
os tokenizadores (não carrega os pesos dos modelos) — roda rápido, sem GPU.

Uso:
    python analyze_subtokens.py --data data/dev_ptbr_corrigido.json
"""

import re
import json
import argparse
from collections import Counter

from transformers import AutoTokenizer

import dataloader


MODELS = {
    "bertimbau_base": "neuralmind/bert-base-portuguese-cased",
    "norberto_base": "Itau-Unibanco/NorBERTo-base",
    "albertina_900m": "PORTULAN/albertina-900m-portuguese-ptbr-encoder",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dev_ptbr_corrigido.json")
    parser.add_argument("--output", default="subtoken_distribution.json")
    return parser.parse_args()


def _fragment_subtoken_count(context_pt: str, sentence_pt: str, tokenizer) -> int:
    """
    Mesma lógica de alinhamento prefixo/sufixo usada em eval_model.py para
    localizar o termo-alvo na sentença tokenizada — aqui só para contar
    quantos subtokens ele ocupa, sem construir nenhum input mascarado.
    """
    ctx = re.sub(r"\s+", " ", context_pt).strip()
    sent = re.sub(r"\s+", " ", sentence_pt).strip()
    blank_pos = ctx.upper().find("BLANK")
    if blank_pos == -1:
        return None

    prefix_text = ctx[:blank_pos].rstrip()
    suffix_text = ctx[blank_pos + len("BLANK"):].lstrip()

    full_ids = tokenizer.encode(sent, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False) if prefix_text else []
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False) if suffix_text else []

    n_pre = len(prefix_ids)
    n_suf = len(suffix_ids)
    frag_len = len(full_ids) - n_pre - n_suf

    if frag_len <= 0:
        return None
    return frag_len


def main():
    args = parse_args()
    dataset = dataloader.StereoSet(args.data)
    examples = dataset.get_intrasentence_examples()

    report = {}

    for nome, model_id in MODELS.items():
        print(f"\nTokenizador: {nome} ({model_id})")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        counts = []
        failures = 0
        for example in examples:
            for sent in example.sentences:
                n = _fragment_subtoken_count(example.context, sent.sentence, tokenizer)
                if n is None:
                    failures += 1
                    continue
                counts.append(n)

        hist = Counter(counts)
        total = len(counts)
        mean = sum(counts) / total if total else 0.0
        pct_multi = 100.0 * sum(v for k, v in hist.items() if k > 1) / total if total else 0.0

        stats = {
            "n_exemplos": total,
            "falhas_alinhamento": failures,
            "media_subtokens": round(mean, 3),
            "max_subtokens": max(counts) if counts else 0,
            "pct_mais_de_1_subtoken": round(pct_multi, 2),
            "histograma": dict(sorted(hist.items())),
        }
        report[nome] = stats

        print(f"  Média de subtokens por termo-alvo : {stats['media_subtokens']}")
        print(f"  % de termos com mais de 1 subtoken : {stats['pct_mais_de_1_subtoken']}%")
        print(f"  Histograma (nº subtokens -> contagem): {stats['histograma']}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório salvo em: {args.output}")


if __name__ == "__main__":
    main()
