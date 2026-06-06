# StereoSet pt-BR

Reprodução do benchmark [StereoSet](https://github.com/moinnadeem/stereoset) para modelos de linguagem em português, usando o dataset traduzido (`ptbr_llm.csv`).

## Estrutura

```
stereoset-ptbr/
├── convert_csv_to_json.py   # Converte o CSV traduzido para JSON no formato StereoSet
├── eval_model.py            # Avalia qualquer modelo MLM em português (MLM + PLL)
├── evaluation.py            # Calcula LM Score, SS Score, ICAT Score
├── dataloader.py            # Lê o JSON gerado
├── requirements.txt
└── data/
    └── dev_ptbr.json        # Gerado por convert_csv_to_json.py
└── predictions/
    └── predictions_*.json   # Gerados por eval_model.py
```

## Modelos suportados

| Modelo | ID HuggingFace | Arquitetura | token_type_ids |
|--------|---------------|-------------|----------------|
| BERTimbau base | `neuralmind/bert-base-portuguese-cased` | BERT | ✅ |
| BERTimbau large | `neuralmind/bert-large-portuguese-cased` | BERT | ✅ |
| Albertina | `PORTULAN/albertina-900m-portuguese-ptbr-encoder` | DeBERTa-v2 | ❌ |
| NorBERTo base | `Itau-Unibanco/NorBERTo-base` | ModernBERT | ❌ |
| NorBERTo large | `Itau-Unibanco/NorBERTo-large` | ModernBERT | ❌ |

O script detecta automaticamente a arquitetura via `model.config.type_vocab_size` e adapta o forward pass.

## Instalação

```bash
pip install -r requirements.txt
```

## Uso passo a passo

### 1. Converter o CSV para JSON

```bash
python convert_csv_to_json.py \
    --input ../ptbr_llm.csv \
    --output data/dev_ptbr.json
```

### 2. Avaliar os modelos

O script `eval_model.py` aceita qualquer modelo MLM em português. A arquitetura é detectada automaticamente.

```bash
# BERTimbau base
python eval_model.py \
    --model neuralmind/bert-base-portuguese-cased \
    --data data/dev_ptbr.json \
    --output predictions/predictions_bertimbau_base.json

# Albertina
python eval_model.py \
    --model PORTULAN/albertina-900m-portuguese-ptbr-encoder \
    --data data/dev_ptbr.json \
    --output predictions/predictions_albertina.json

# NorBERTo base
python eval_model.py \
    --model Itau-Unibanco/NorBERTo-base \
    --data data/dev_ptbr.json \
    --output predictions/predictions_norberto_base.json

# Somente intrasentence (mais rápido)
python eval_model.py --model <id> --skip-intersentence ...

# Sem GPU
python eval_model.py --model <id> --no-cuda ...
```

### 3. Calcular as métricas

```bash
# Um modelo específico
python evaluation.py \
    --data data/dev_ptbr.json \
    --predictions predictions/predictions_bertimbau_base.json

# Todos os modelos em predictions/
python evaluation.py \
    --data data/dev_ptbr.json \
    --predictions-dir predictions/ \
    --output results.json
```

## Métricas

| Métrica | Descrição | Valor ideal |
|---------|-----------|-------------|
| **LM Score** | % de vezes que o modelo prefere frases relacionadas às não-relacionadas | Alto (próximo de 100) |
| **SS Score** | % de vezes que o modelo prefere o estereótipo ao anti-estereótipo | ~50 (sem viés) |
| **ICAT Score** | Combinação: `LM × min(SS, 100-SS) / 50` | Alto (próximo de 100) |

## Referência

Nadeem et al. (2020). [StereoSet: Measuring stereotypical bias in pretrained language models](https://arxiv.org/abs/2004.09456).
