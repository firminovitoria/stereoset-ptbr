"""
Avalia qualquer modelo MLM em português no benchmark StereoSet pt-BR.

Tarefas:
  - Intrasentence: MLM — probabilidade do token correto no lugar de [MASK]
  - Intersentence: pseudo-log-likelihood (PLL) — log P de cada token da sentença
    candidata mascarado individualmente, dado o contexto (Salazar et al., 2020)
    https://arxiv.org/abs/1910.14659

Arquiteturas suportadas (detectadas automaticamente via config do modelo):
  - BERT         neuralmind/bert-base-portuguese-cased   (BERTimbau base)
                 neuralmind/bert-large-portuguese-cased  (BERTimbau large)
  - DeBERTa-v2   PORTULAN/albertina-900m-portuguese-ptbr-encoder  (Albertina)
  - ModernBERT   Itau-Unibanco/NorBERTo-base             (NorBERTo base)

A diferença principal entre arquiteturas: BERT usa token_type_ids (type_vocab_size=2)
para distinguir o par contexto/sentença; DeBERTa-v2 e ModernBERT não (type_vocab_size=0).
O script detecta isso via model.config e adapta o forward pass automaticamente.

Uso:
    python eval_model.py \\
        --model neuralmind/bert-base-portuguese-cased \\
        --data data/dev_ptbr.json \\
        --output predictions/predictions_bertimbau_base.json

    python eval_model.py \\
        --model PORTULAN/albertina-900m-portuguese-ptbr-encoder \\
        --data data/dev_ptbr.json \\
        --output predictions/predictions_albertina.json

    python eval_model.py \\
        --model Itau-Unibanco/NorBERTo-base \\
        --data data/dev_ptbr.json \\
        --output predictions/predictions_norberto_base.json

    # Somente intrasentence (mais rápido):
    python eval_model.py --model <id> --skip-intersentence ...

    # Sem GPU:
    python eval_model.py --model <id> --no-cuda ...
"""

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
    """
    Retorna True se o modelo suporta token_type_ids (type_vocab_size > 1).
    BERT: type_vocab_size=2  → True
    DeBERTa-v2, ModernBERT: type_vocab_size=0 → False
    """
    return getattr(model.config, "type_vocab_size", 0) > 1


# ---------------------------------------------------------------------------
# Intrasentence: MLM scoring
# ---------------------------------------------------------------------------

def _build_masked_input(context_pt: str, sentence_pt: str, tokenizer, max_length: int):
    """
    Constrói input com [MASK] no lugar de BLANK, usando a API tokenizer() diretamente.

    Evita inserir cls_token_id / sep_token_id manualmente (que podem ser None em
    tokenizers ModernBERT). O tokenizer() adiciona os tokens especiais corretos
    para cada arquitetura automaticamente.

    Para os token-alvo, tokeniza a sentença completa e extrai o fragmento por
    alinhamento de prefixo — necessário com SentencePiece (Albertina/DeBERTa-v2)
    pois o marcador ▁ (word boundary) muda quando o fragmento é tokenizado em
    isolamento vs. em contexto.

    Retorna (input_ids_tensor, mask_positions, target_ids).
    """
    ctx = re.sub(r"\s+", " ", context_pt).strip()
    sent = re.sub(r"\s+", " ", sentence_pt).strip()
    blank_pos = ctx.upper().find("BLANK")

    if blank_pos != -1:
        masked_text = ctx[:blank_pos] + tokenizer.mask_token + ctx[blank_pos + len("BLANK"):]
    else:
        masked_text = ctx  # sem BLANK: não haverá posições mascaradas

    # tokenizer() adiciona CLS/SEP corretamente para qualquer arquitetura
    enc = tokenizer(
        masked_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids_list = enc["input_ids"][0].tolist()
    mask_positions = [i for i, t in enumerate(ids_list) if t == tokenizer.mask_token_id]

    if not mask_positions:
        return enc["input_ids"], [], []

    # IDs alvo: tokeniza a sentença completa e alinha pelo prefixo.
    # Tokenizando em contexto (sentença inteira) o SentencePiece produz IDs
    # consistentes com o que o modelo verá no input mascarado.
    full_ids = tokenizer.encode(sent, add_special_tokens=False)

    if blank_pos != -1:
        prefix_text = ctx[:blank_pos].rstrip()
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False) if prefix_text else []
        n_pre = len(prefix_ids)
        frag_ids = full_ids[n_pre: n_pre + len(mask_positions)]
        # Fallback se o alinhamento falhou (diferença de boundary SentencePiece)
        if not frag_ids:
            frag_ids = full_ids[:len(mask_positions)]
    else:
        frag_ids = full_ids[:len(mask_positions)]

    n = min(len(mask_positions), len(frag_ids))
    return enc["input_ids"], mask_positions[:n], frag_ids[:n]


def evaluate_intrasentence(tokenizer, model, dataset: dataloader.StereoSet,
                           device: str, max_length: int) -> list:
    vocab_size = model.config.vocab_size  # para bounds check
    results = []
    for example in tqdm(dataset.get_intrasentence_examples(), desc="Intrasentence"):
        for sent in example.sentences:
            input_ids, mask_positions, target_ids = _build_masked_input(
                example.context, sent.sentence, tokenizer, max_length
            )

            if not mask_positions:
                results.append({"id": sent.ID, "score": 0.0})
                continue

            with torch.no_grad():
                logits = model(input_ids.to(device)).logits  # (1, seq_len, vocab)

            log_probs = torch.log_softmax(logits[0], dim=-1)
            seq_len = log_probs.shape[0]

            token_log_probs = [
                log_probs[pos, tid].item()
                for pos, tid in zip(mask_positions, target_ids)
                # guarda contra pos fora da seq e tid fora do vocab
                if pos < seq_len and 0 <= tid < vocab_size
            ]

            score = float(np.exp(np.mean(token_log_probs))) if token_log_probs else 0.0
            results.append({"id": sent.ID, "score": score})

    return results


# ---------------------------------------------------------------------------
# Intersentence: pseudo-log-likelihood (PLL) scoring
#
# Formato do input:
#   Com token_type_ids (BERT):
#     [CLS] contexto [SEP] sent_prefix [MASK] sent_suffix [SEP]
#      ^^^^ seg 0 ^^^^^^^^              ^^^^ seg 1 ^^^^^^^^
#
#   Sem token_type_ids (DeBERTa-v2, ModernBERT):
#     [CLS] contexto [SEP] sent_prefix [MASK] sent_suffix [SEP]
#     (sem distinção de segmentos — o modelo usa atenção relativa)
# ---------------------------------------------------------------------------

def _pll_score(context: str, sentence: str, tokenizer, model, device: str,
               max_length: int, use_tti: bool) -> float:
    """
    Pseudo-log-likelihood de `sentence` dado `context`.
    Mascara um token de `sentence` por vez e acumula log P do token correto.
    Retorna a média das log-probs (normalizada pelo comprimento).

    Usa tokenizer(context, sentence) para construir o par — evita inserir
    cls_token_id / sep_token_id manualmente (problema em tokenizers ModernBERT).

    Estratégia para encontrar as posições da sentença no encoding combinado:
    - BERT (use_tti=True): token_type_ids == 1 marca a sentença candidata.
    - DeBERTa-v2 / ModernBERT (use_tti=False): truncation="only_first" garante
      que a sentença esteja intacta; as últimas N posições não-especiais são dela.
    """
    sent_ids = tokenizer.encode(sentence, add_special_tokens=False)
    if not sent_ids:
        return 0.0

    vocab_size = model.config.vocab_size

    # Encoding do par completo — tokenizer() adiciona CLS/SEP corretamente
    # truncation="only_first" preserva a sentença candidata intacta
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

    # Posições dos tokens da sentença no encoding combinado
    if use_tti and "token_type_ids" in base_enc:
        # BERT: segment 1 = sentença candidata
        tti = base_enc["token_type_ids"][0].tolist()
        sent_positions = [j for j, t in enumerate(tti) if t == 1]
    else:
        # Sem token_type_ids: as últimas N posições não-especiais são da sentença
        # (truncation="only_first" garante que sent_ids está intacto no final)
        special_ids = set(tokenizer.all_special_ids)
        all_non_special = [j for j, t in enumerate(base_ids) if t not in special_ids]
        n = min(len(sent_ids), len(all_non_special))
        sent_positions = all_non_special[-n:]

    log_probs = []
    for i, target_id in enumerate(sent_ids):
        if i >= len(sent_positions) or not (0 <= target_id < vocab_size):
            continue

        # Mascara apenas a posição i — reutiliza base_ids sem reconstruir do zero
        ids_t = torch.tensor([base_ids]).to(device)
        ids_t[0, sent_positions[i]] = tokenizer.mask_token_id

        kwargs = {}
        if use_tti and "token_type_ids" in base_enc:
            kwargs["token_type_ids"] = base_enc["token_type_ids"].to(device)

        with torch.no_grad():
            logits = model(ids_t, **kwargs).logits  # (1, seq, vocab)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
