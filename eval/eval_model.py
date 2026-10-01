import json
import re
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

import dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="neuralmind/bert-base-portuguese-cased",
        help="ID do modelo no HuggingFace Hub.",
    )
    parser.add_argument("--data", default="data/dev_ptbr.json")
    parser.add_argument("--output", default=None,
                        help="Arquivo de saída. Padrão: predictions/predictions_<slug>.json")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--skip-intrasentence", action="store_true", default=False)
    parser.add_argument("--skip-intersentence", action="store_true", default=False)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def _uses_token_type_ids(model) -> bool:
    return getattr(model.config, "type_vocab_size", 0) > 1


def _target_span(context_pt: str, sentence_pt: str, tokenizer):
    ctx = re.sub(r"\s+", " ", context_pt).strip()
    sent = re.sub(r"\s+", " ", sentence_pt).strip()
    blank_pos = ctx.upper().find("BLANK")
    if blank_pos == -1:
        return None, None, None

    prefix_text = ctx[:blank_pos].rstrip()
    suffix_text = ctx[blank_pos + len("BLANK"):].lstrip()

    full_ids = tokenizer.encode(sent, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False) if prefix_text else []
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False) if suffix_text else []

    n_pre = len(prefix_ids)
    n_suf = len(suffix_ids)
    frag_end = len(full_ids) - n_suf

    if n_pre >= frag_end or n_pre > len(full_ids):
        n_pre = min(n_pre, len(full_ids))
        frag_end = len(full_ids)

    return sent, n_pre, frag_end


def _score_intrasentence(context_pt: str, sentence_pt: str, tokenizer, model,
                          device: str, max_length: int) -> float:
    sent, n_pre, frag_end = _target_span(context_pt, sentence_pt, tokenizer)
    if sent is None:
        return 0.0

    enc = tokenizer(
        sent,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    base_ids = enc["input_ids"][0].tolist()
    special_mask = enc["special_tokens_mask"][0].tolist()
    vocab_size = model.config.vocab_size

    offset = 0
    while offset < len(special_mask) and special_mask[offset] == 1:
        offset += 1

    span_start = offset + n_pre
    span_end = min(offset + frag_end, len(base_ids))

    log_probs = []
    for pos in range(span_start, span_end):
        target_id = base_ids[pos]
        if not (0 <= target_id < vocab_size):
            continue

        ids_t = torch.tensor([base_ids]).to(device)
        ids_t[0, pos] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(ids_t).logits

        lp = torch.log_softmax(logits[0, pos], dim=-1)[target_id].item()
        log_probs.append(lp)

    return float(np.exp(np.mean(log_probs))) if log_probs else 0.0


def evaluate_intrasentence(tokenizer, model, dataset: dataloader.StereoSet,
                           device: str, max_length: int) -> list:
    results = []
    for example in tqdm(dataset.get_intrasentence_examples(), desc="Intrasentence"):
        for sent in example.sentences:
            score = _score_intrasentence(
                example.context, sent.sentence, tokenizer, model, device, max_length
            )
            results.append({"id": sent.ID, "score": score})
    return results


def _pll_score(context: str, sentence: str, tokenizer, model, device: str,
               max_length: int, use_tti: bool) -> float:
    sent_ids = tokenizer.encode(sentence, add_special_tokens=False)
    if not sent_ids:
        return 0.0

    vocab_size = model.config.vocab_size

    if context:
        base_enc = tokenizer(
            context, sentence,
            add_special_tokens=True,
            max_length=max_length,
            truncation="only_first",
            return_tensors="pt",
        )
    else:
        base_enc = tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

    base_ids = base_enc["input_ids"][0].tolist()

    if use_tti and "token_type_ids" in base_enc:
        tti = base_enc["token_type_ids"][0].tolist()
        sent_positions = [j for j, t in enumerate(tti) if t == 1]
    else:
        special_ids = set(tokenizer.all_special_ids)
        all_non_special = [j for j, t in enumerate(base_ids) if t not in special_ids]
        n = min(len(sent_ids), len(all_non_special))
        sent_positions = all_non_special[-n:]

    log_probs = []
    for i, target_id in enumerate(sent_ids):
        if i >= len(sent_positions) or not (0 <= target_id < vocab_size):
            continue

        ids_t = torch.tensor([base_ids]).to(device)
        ids_t[0, sent_positions[i]] = tokenizer.mask_token_id

        kwargs = {}
        if use_tti and "token_type_ids" in base_enc:
            kwargs["token_type_ids"] = base_enc["token_type_ids"].to(device)

        with torch.no_grad():
            logits = model(ids_t, **kwargs).logits

        mask_pos = sent_positions[i]
        lp = torch.log_softmax(logits[0, mask_pos], dim=-1)[target_id].item()
        log_probs.append(lp)

    return float(np.mean(log_probs)) if log_probs else 0.0


def evaluate_intersentence(tokenizer, model, dataset: dataloader.StereoSet,
                            device: str, max_length: int, use_tti: bool) -> list:
    results = []
    for example in tqdm(dataset.get_intersentence_examples(), desc="Intersentence (PLL)"):
        for sent in example.sentences:
            score = _pll_score(
                example.context, sent.sentence,
                tokenizer, model, device, max_length, use_tti,
            )
            results.append({"id": sent.ID, "score": score})
    return results


def _output_path(model_id: str) -> str:
    slug = model_id.replace("/", "_").replace("-", "_").lower()
    return f"predictions/predictions_{slug}.json"


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    output = args.output or _output_path(args.model)

    print(f"Modelo   : {args.model}")
    print(f"Dispositivo: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()

    if tokenizer.mask_token_id is None:
        raise ValueError(
            f"O tokenizer de '{args.model}' não define mask_token. "
            "Verifique se o modelo suporta Masked Language Modeling (MLM). "
            "Consulte o model card no HuggingFace para confirmar a task."
        )

    use_tti = _uses_token_type_ids(model)
    arch = model.config.model_type
    print(f"Arquitetura: {arch}  |  token_type_ids: {use_tti}")

    dataset = dataloader.StereoSet(args.data)
    result: dict = {}

    if not args.skip_intrasentence:
        print("\n[Intrasentence] MLM scoring...")
        result["intrasentence"] = evaluate_intrasentence(
            tokenizer, model, dataset, device, args.max_length
        )

    if not args.skip_intersentence:
        print("\n[Intersentence] PLL scoring...")
        result["intersentence"] = evaluate_intersentence(
            tokenizer, model, dataset, device, args.max_length, use_tti
        )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nPredições salvas em: {output}")


if __name__ == "__main__":
    main()
