"""
Avalia BERTimbau (ou qualquer BERT em português) no benchmark StereoSet pt-BR.

Tarefas:
  - Intrasentence: MLM — probabilidade do token correto no lugar de [MASK]
  - Intersentence: pseudo-log-likelihood (PLL) — soma de log P de cada token da
    sentença candidata mascarado individualmente, dado o contexto (Salazar et al., 2020)

Modelos testados:
  - neuralmind/bert-base-portuguese-cased   (BERTimbau base)
  - neuralmind/bert-large-portuguese-cased  (BERTimbau large)

Uso:
    python eval_bertimbau.py \\
        --model neuralmind/bert-base-portuguese-cased \\
        --data data/dev_ptbr.json \\
        --output predictions/predictions_bertimbau_base.json

    # Só intrasentence (mais rápido):
    python eval_bertimbau.py --skip-intersentence ...

    # Sem GPU:
    python eval_bertimbau.py --no-cuda ...
"""

import json
import re
import argparse
from collections import defaultdict

import numpy as np
import torch
from transformers import BertTokenizerFast, BertForMaskedLM
from tqdm import tqdm

import dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="neuralmind/bert-base-portuguese-cased",
                        help="Modelo HuggingFace a avaliar.")
    parser.add_argument("--data", default="data/dev_ptbr.json",
                        help="Dataset StereoSet pt-BR em JSON.")
    parser.add_argument("--output", default="predictions/predictions_bertimbau.json",
                        help="Arquivo de saída com os scores.")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--skip-intrasentence", action="store_true", default=False)
    parser.add_argument("--skip-intersentence", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Intrasentence: MLM scoring
# ---------------------------------------------------------------------------

def _find_inserted_tokens(context_pt: str, sentence_pt: str, tokenizer) -> list[int]:
    """
    Descobre quais token-ids foram inseridos no lugar de BLANK.

    Estratégia:
      1. Substitui BLANK por [MASK] no contexto.
      2. Tokeniza contexto-com-mask e sentença-completa.
      3. Retorna os ids dos tokens na sentença que estão na posição de [MASK].
    """
    # Normaliza espaços
    ctx = re.sub(r"\s+", " ", context_pt).strip()
    sent = re.sub(r"\s+", " ", sentence_pt).strip()

    # Extrai o trecho que foi inserido no lugar de BLANK
    blank_pos = ctx.upper().find("BLANK")
    if blank_pos == -1:
        # contexto sem BLANK (intersentence misturado): fallback
        tokens = tokenizer.encode(sent, add_special_tokens=False)
        return tokens

    prefix = ctx[:blank_pos].strip()
    suffix = ctx[blank_pos + len("BLANK"):].strip()

    # Remove prefix/suffix da sentença para isolar o fragmento inserido
    fragment = sent
    if prefix and fragment.lower().startswith(prefix.lower()):
        fragment = fragment[len(prefix):].strip()
    if suffix and fragment.lower().endswith(suffix.lower()):
        fragment = fragment[: len(fragment) - len(suffix)].strip()

    token_ids = tokenizer.encode(fragment, add_special_tokens=False)
    return token_ids if token_ids else tokenizer.encode(sent, add_special_tokens=False)


def _build_masked_input(context_pt: str, token_ids: list[int], tokenizer, max_length: int):
    """
    Constrói input com [MASK] x N no lugar de BLANK.
    Retorna (input_ids_tensor, mask_positions).
    """
    ctx = re.sub(r"\s+", " ", context_pt).strip()
    blank_pos = ctx.upper().find("BLANK")

    if blank_pos != -1:
        prefix = ctx[:blank_pos].strip()
        suffix = ctx[blank_pos + len("BLANK"):].strip()

        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
        mask_ids = [tokenizer.mask_token_id] * len(token_ids)

        ids = (
            [tokenizer.cls_token_id]
            + prefix_ids
            + mask_ids
            + suffix_ids
            + [tokenizer.sep_token_id]
        )
        mask_positions = list(range(1 + len(prefix_ids), 1 + len(prefix_ids) + len(mask_ids)))
    else:
        # Fallback: mascara todos os tokens da sentença
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
        mask_ids = [tokenizer.mask_token_id] * len(token_ids)
        ids = [tokenizer.cls_token_id] + ctx_ids + mask_ids + [tokenizer.sep_token_id]
        mask_positions = list(range(1 + len(ctx_ids), 1 + len(ctx_ids) + len(mask_ids)))

    ids = ids[:max_length]
    mask_positions = [p for p in mask_positions if p < max_length]

    input_tensor = torch.tensor([ids])
    return input_tensor, mask_positions, token_ids[: len(mask_positions)]


def evaluate_intrasentence(model_name: str, dataset: dataloader.StereoSet,
                           device: str, max_length: int) -> list[dict]:
    print("\n[Intrasentence] Carregando modelo MLM...")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    results = []

    examples = dataset.get_intrasentence_examples()
    for example in tqdm(examples, desc="Intrasentence"):
        for sent in example.sentences:
            token_ids = _find_inserted_tokens(example.context, sent.sentence, tokenizer)
            input_ids, mask_positions, target_ids = _build_masked_input(
                example.context, token_ids, tokenizer, max_length
            )

            if not mask_positions:
                results.append({"id": sent.ID, "score": 0.0})
                continue

            input_ids = input_ids.to(device)
            with torch.no_grad():
                logits = model(input_ids).logits  # (1, seq_len, vocab)

            log_probs = torch.log_softmax(logits[0], dim=-1)

            token_log_probs = []
            for pos, tid in zip(mask_positions, target_ids):
                if pos < log_probs.shape[0]:
                    token_log_probs.append(log_probs[pos, tid].item())

            score = float(np.exp(np.mean(token_log_probs))) if token_log_probs else 0.0
            results.append({"id": sent.ID, "score": score})

    return results


# ---------------------------------------------------------------------------
# Intersentence: pseudo-log-likelihood (PLL) scoring
#
# Para cada token t_i da sentença candidata, mascara t_i e soma
# log P(t_i | contexto + sentença com t_i mascarado).
# Método: Salazar et al., 2020 — "Masked Language Model Scoring"
#         https://arxiv.org/abs/1910.14659
#
# Formato do input para o BERT:
#   [CLS] contexto [SEP] sent_prefix [MASK] sent_suffix [SEP]
# ---------------------------------------------------------------------------

def _pll_score(context: str, sentence: str, tokenizer, model, device: str,
               max_length: int) -> float:
    """
    Calcula a pseudo-log-likelihood de `sentence` dado `context`.
    Mascara um token de `sentence` por vez e acumula log P do token correto.
    Retorna a média das log-probs (normalizada pelo comprimento).
    """
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    sent_ids = tokenizer.encode(sentence, add_special_tokens=False)

    # Trunca o contexto se o par ultrapassar max_length
    # Estrutura: [CLS] ctx [SEP] sent [SEP]  →  2 + len(ctx) + 1 + len(sent) + 1
    overhead = 3  # [CLS], [SEP], [SEP]
    max_ctx = max_length - overhead - len(sent_ids)
    if max_ctx <= 0:
        # Sentença candidata sozinha já ocupa o limite; avalia sem contexto
        ctx_ids = []
        max_ctx = 0
    ctx_ids = ctx_ids[-max_ctx:] if max_ctx else []  # mantém final do contexto

    log_probs = []
    for i, target_id in enumerate(sent_ids):
        # Constrói sequência com o i-ésimo token da sentença mascarado
        masked_sent = sent_ids[:i] + [tokenizer.mask_token_id] + sent_ids[i + 1:]

        if ctx_ids:
            input_ids = (
                [tokenizer.cls_token_id]
                + ctx_ids
                + [tokenizer.sep_token_id]
                + masked_sent
                + [tokenizer.sep_token_id]
            )
            # token_type_ids: 0 para contexto, 1 para sentença candidata
            token_type_ids = (
                [0] * (1 + len(ctx_ids) + 1)
                + [1] * (len(masked_sent) + 1)
            )
        else:
            input_ids = (
                [tokenizer.cls_token_id]
                + masked_sent
                + [tokenizer.sep_token_id]
            )
            token_type_ids = [0] * len(input_ids)

        mask_pos = input_ids.index(tokenizer.mask_token_id)

        ids_tensor = torch.tensor([input_ids]).to(device)
        tti_tensor = torch.tensor([token_type_ids]).to(device)

        with torch.no_grad():
            logits = model(ids_tensor, token_type_ids=tti_tensor).logits  # (1, seq, vocab)

        log_prob = torch.log_softmax(logits[0, mask_pos], dim=-1)[target_id].item()
        log_probs.append(log_prob)

    return float(np.mean(log_probs)) if log_probs else 0.0


def evaluate_intersentence(model_name: str, dataset: dataloader.StereoSet,
                            device: str, max_length: int) -> list[dict]:
    print("\n[Intersentence] Carregando modelo MLM para PLL scoring...")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    results = []

    examples = dataset.get_intersentence_examples()
    for example in tqdm(examples, desc="Intersentence (PLL)"):
        for sent in example.sentences:
            score = _pll_score(
                example.context, sent.sentence,
                tokenizer, model, device, max_length,
            )
            results.append({"id": sent.ID, "score": score})

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Dispositivo: {device}")
    print(f"Modelo: {args.model}")

    dataset = dataloader.StereoSet(args.data)
    output: dict = {}

    if not args.skip_intrasentence:
        output["intrasentence"] = evaluate_intrasentence(
            args.model, dataset, device, args.max_length
        )

    if not args.skip_intersentence:
        output["intersentence"] = evaluate_intersentence(
            args.model, dataset, device, args.max_length
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nPredições salvas em: {args.output}")


if __name__ == "__main__":
    main()
