"""
Avalia modelos autorregressivos (causal LM) em português no benchmark StereoSet pt-BR.

Complementa eval_model.py (MLM/PLL) com a contribuição 2(b) sugerida na banca de
qualificação: modelos de linguagem mascarada não são o único tipo de modelo aberto
disponível em português, e o StereoSet não exige MLM — exige apenas acesso às
probabilidades de sequência, que qualquer causal LM expõe nativamente.

Tarefas:
  - Intrasentence: log-likelihood do termo-alvo, condicionado ao prefixo do contexto,
    via teacher forcing em um único forward pass (sem mascaramento).
  - Intersentence: log-likelihood de sequência da sentença candidata inteira,
    condicionada à sentença de contexto, também via um único forward pass.

Isso é mais barato computacionalmente que a PLL usada para os modelos MLM (que exige
N forward passes por sentença, um por token mascarado) e evita o problema de
subtokens discutido no Capítulo 3: não há necessidade de mascarar; o modelo lê a
sequência inteira de uma vez e a probabilidade de cada subtoken do termo-alvo é
tomada diretamente da distribuição prevista na posição anterior.

Modelos alvo (mas funciona com qualquer causal LM no HuggingFace Hub):
  - TucanoBR/Tucano-1b1                 (Tucano, Apache 2.0, ~1.1B)
  - maritaca-ai/sabia-7b                (Sabiá-7B, licença estilo LLaMA-1, ~7B)

Uso:
    python eval_autoregressive.py \\
        --model TucanoBR/Tucano-1b1 \\
        --data data/dev_ptbr.json \\
        --output predictions/predictions_tucano_1b1.json

    python eval_autoregressive.py \\
        --model maritaca-ai/sabia-7b \\
        --data data/dev_ptbr.json \\
        --output predictions/predictions_sabia_7b.json \\
        --load-in-8bit

    # Somente intrasentence (mais rápido):
    python eval_autoregressive.py --model <id> --skip-intersentence ...

    # Sem GPU:
    python eval_autoregressive.py --model <id> --no-cuda ...

Nota sobre memória (GPU T4, 16GB): Sabiá-7B em float16 ocupa ~14GB só de
pesos, deixando pouca folga para ativações. Use --load-in-8bit (via
bitsandbytes) para reduzir a pegada de memória para ~7GB. Tucano-1b1 roda
confortavelmente em float16 ou até float32 numa T4.
"""

import json
import re
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

import dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="TucanoBR/Tucano-1b1",
        help="ID do modelo causal LM no HuggingFace Hub.",
    )
    parser.add_argument("--data", default="data/dev_ptbr.json")
    parser.add_argument("--output", default=None,
                        help="Arquivo de saída. Padrão: predictions/predictions_<slug>.json")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--skip-intrasentence", action="store_true", default=False)
    parser.add_argument("--skip-intersentence", action="store_true", default=False)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Precisão de carregamento do modelo. Use float16 para modelos grandes (ex: Sabiá-7B).",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        default=False,
        help="Carrega o modelo quantizado em 8-bit via bitsandbytes (requer pip install bitsandbytes accelerate). "
             "Recomendado para Sabiá-7B em GPUs com pouca VRAM (ex: T4 16GB). Ignora --dtype se ativado.",
    )
    return parser.parse_args()


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Utilidades comuns
# ---------------------------------------------------------------------------

def _bos_ids(tokenizer) -> list[int]:
    """Retorna [bos_token_id] se o tokenizer define um, senão []."""
    return [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []


def _sequence_logprobs(seq_ids: list[int], model, device: str):
    """
    Um único forward pass sobre `seq_ids`. Retorna log_softmax de todas as posições.

    logits[t] é a distribuição prevista para o token na posição t+1
    (convenção padrão de causal LM da HuggingFace).
    """
    input_ids = torch.tensor([seq_ids]).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # (seq_len, vocab)
    return torch.log_softmax(logits, dim=-1)


def _score_span(log_probs, seq_ids: list[int], span_start: int, span_end: int, vocab_size: int):
    """
    Log-probs médias (teacher forcing) para seq_ids[span_start:span_end],
    cada uma extraída de log_probs[posição - 1].
    """
    seq_len = log_probs.shape[0]
    vals = []
    for pos in range(span_start, span_end):
        if pos <= 0 or pos - 1 >= seq_len:
            continue
        tid = seq_ids[pos]
        if not (0 <= tid < vocab_size):
            continue
        vals.append(log_probs[pos - 1, tid].item())
    return vals


# ---------------------------------------------------------------------------
# Intrasentence: log-likelihood do termo-alvo via teacher forcing
# ---------------------------------------------------------------------------

def _intrasentence_span(context_pt: str, sentence_pt: str, tokenizer, bos: list[int]):
    """
    Localiza o span de tokens do termo-alvo dentro da tokenização da sentença
    completa, por alinhamento de prefixo/sufixo (evita problemas de fronteira
    de subtoken ao tokenizar o fragmento isoladamente).

    Retorna (seq_ids, span_start, span_end) ou (None, None, None) se BLANK
    não for encontrado no contexto.
    """
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

    # Guarda contra alinhamento inconsistente (fronteiras de SentencePiece/BPE)
    if n_pre >= frag_end or n_pre > len(full_ids):
        n_pre = min(n_pre, len(full_ids))
        frag_end = len(full_ids)

    seq_ids = bos + full_ids
    offset = len(bos)
    return seq_ids, offset + n_pre, offset + frag_end


def evaluate_intrasentence(tokenizer, model, dataset: dataloader.StereoSet,
                           device: str, max_length: int) -> list:
    bos = _bos_ids(tokenizer)
    vocab_size = model.config.vocab_size
    results = []

    for example in tqdm(dataset.get_intrasentence_examples(), desc="Intrasentence (causal LM)"):
        for sent in example.sentences:
            seq_ids, span_start, span_end = _intrasentence_span(
                example.context, sent.sentence, tokenizer, bos
            )

            if seq_ids is None or span_start >= span_end:
                results.append({"id": sent.ID, "score": 0.0})
                continue

            seq_ids = seq_ids[:max_length]
            span_end = min(span_end, max_length)
            if span_start >= span_end:
                results.append({"id": sent.ID, "score": 0.0})
                continue

            log_probs = _sequence_logprobs(seq_ids, model, device)
            token_log_probs = _score_span(log_probs, seq_ids, span_start, span_end, vocab_size)

            score = float(np.exp(np.mean(token_log_probs))) if token_log_probs else 0.0
            results.append({"id": sent.ID, "score": score})

    return results


# ---------------------------------------------------------------------------
# Intersentence: log-likelihood de sequência da sentença candidata inteira,
# condicionada ao contexto — um único forward pass (sem PLL).
# ---------------------------------------------------------------------------

def _intersentence_span(context: str, sentence: str, tokenizer, bos: list[int], max_length: int):
    ctx = re.sub(r"\s+", " ", context).strip()
    sent = re.sub(r"\s+", " ", sentence).strip()

    if ctx:
        joined = ctx + " " + sent
    else:
        joined = sent

    joined_ids = tokenizer.encode(joined, add_special_tokens=False)
    ctx_ids = tokenizer.encode(ctx, add_special_tokens=False) if ctx else []
    n_ctx = min(len(ctx_ids), len(joined_ids))

    # Reserva espaço para a sentença candidata; trunca o contexto pela esquerda
    # se necessário (mesmo espírito do truncation="only_first" usado na PLL).
    overhead = len(bos)
    sent_len = len(joined_ids) - n_ctx
    max_ctx = max_length - overhead - sent_len
    if max_ctx < 0:
        # sentença sozinha já estoura o limite: mantém só o final dela
        joined_ids = joined_ids[-(max_length - overhead):]
        n_ctx = 0
    elif n_ctx > max_ctx:
        drop = n_ctx - max_ctx
        joined_ids = joined_ids[drop:]
        n_ctx = max_ctx

    seq_ids = bos + joined_ids
    offset = len(bos)
    return seq_ids, offset + n_ctx, len(seq_ids)


def evaluate_intersentence(tokenizer, model, dataset: dataloader.StereoSet,
                            device: str, max_length: int) -> list:
    bos = _bos_ids(tokenizer)
    vocab_size = model.config.vocab_size
    results = []

    for example in tqdm(dataset.get_intersentence_examples(), desc="Intersentence (seq. log-likelihood)"):
        for sent in example.sentences:
            seq_ids, span_start, span_end = _intersentence_span(
                example.context, sent.sentence, tokenizer, bos, max_length
            )

            if span_start >= span_end:
                results.append({"id": sent.ID, "score": 0.0})
                continue

            log_probs = _sequence_logprobs(seq_ids, model, device)
            token_log_probs = _score_span(log_probs, seq_ids, span_start, span_end, vocab_size)

            score = float(np.mean(token_log_probs)) if token_log_probs else 0.0
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

    print(f"Modelo     : {args.model}")
    print(f"Dispositivo: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.load_in_8bit:
        if device != "cuda":
            raise ValueError("--load-in-8bit requer GPU (bitsandbytes não roda em CPU).")
        print("Precisão   : int8 (bitsandbytes)")
        # device_map carrega o modelo direto na GPU; NÃO chamar .to(device) em
        # modelos quantizados via bitsandbytes (não é suportado).
        # Versões recentes do transformers exigem BitsAndBytesConfig em vez do
        # kwarg legado load_in_8bit=True direto no from_pretrained.
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant_config, device_map={"": 0}
        )
    else:
        print(f"Precisão   : {args.dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=_DTYPE_MAP[args.dtype]
        ).to(device)
    model.eval()

    dataset = dataloader.StereoSet(args.data)
    result: dict = {}

    if not args.skip_intrasentence:
        print("\n[Intrasentence] log-likelihood do termo-alvo (teacher forcing)...")
        result["intrasentence"] = evaluate_intrasentence(
            tokenizer, model, dataset, device, args.max_length
        )

    if not args.skip_intersentence:
        print("\n[Intersentence] log-likelihood de sequência...")
        result["intersentence"] = evaluate_intersentence(
            tokenizer, model, dataset, device, args.max_length
        )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nPredições salvas em: {output}")


if __name__ == "__main__":
    main()
