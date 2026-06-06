# StereoSet-PTBR

StereoSet-PTBR é uma adaptação do benchmark StereoSet para o português brasileiro, projetada para avaliar viés estereotípico em modelos de linguagem pré-treinados. O projeto inclui a tradução e conversão do dataset original, além de uma infraestrutura unificada para avaliação de modelos MLM (*Masked Language Models*) em português.

O benchmark permite medir associações estereotípicas em diferentes grupos sociais por meio das métricas propostas por Nadeem et al. (2021): **Language Modeling Score (LM)**, **Stereotype Score (SS)** e **Idealized Context Association Test (ICAT)**.

## Objetivos

* Adaptar o StereoSet para o português brasileiro.
* Avaliar viés social em modelos de linguagem treinados para português.
* Fornecer um pipeline reproduzível para comparação entre arquiteturas.
* Servir como base para pesquisas em justiça algorítmica, viés e mitigação em modelos de linguagem.

---

## Estrutura do Projeto

```text
stereoset-ptbr/
├── convert_csv_to_json.py      # Converte o dataset traduzido para o formato StereoSet
├── eval_model.py              # Avaliação de modelos MLM
├── evaluation.py              # Cálculo das métricas LM, SS e ICAT
├── dataloader.py              # Carregamento do dataset
├── requirements.txt
├── README.md
│
├── data/
│   └── dev_ptbr.json          # Dataset no formato StereoSet
│
├── predictions/
│   ├── predictions_*.json     # Predições geradas pelos modelos
│
├── results/
│   └── results.json           # Resultados agregados
│
└── notebooks/
    └── StereoSet-PTBR: Avaliação de Viés em Modelos de Linguagem para Português.ipynb   # Notebook de execução dos experimentos
```

---

## Modelos Suportados

O pipeline foi desenvolvido para funcionar com qualquer modelo compatível com `AutoModelForMaskedLM` da biblioteca Transformers.

| Modelo              | Hugging Face ID                                   | Arquitetura |
| ------------------- | ------------------------------------------------- | ----------- |
| BERTimbau Base      | `neuralmind/bert-base-portuguese-cased`           | BERT        |
| NorBERTo Base       | `Itau-Unibanco/NorBERTo-base`                     | ModernBERT  |
| Albertina 900M PTBR | `PORTULAN/albertina-900m-portuguese-ptbr-encoder` | DeBERTa-v2  |

O sistema detecta automaticamente características da arquitetura, como suporte a `token_type_ids`, permitindo a avaliação de diferentes famílias de modelos sem alterações no código.

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Fluxo Experimental

### 1. Converter o Dataset

Converta o dataset traduzido para o formato JSON compatível com o StereoSet.

```bash
python convert_csv_to_json.py \
    --input ptbr_llm.csv \
    --output data/dev_ptbr.json
```

### 2. Executar a Avaliação

Exemplo utilizando o BERTimbau Base:

```bash
python eval_model.py \
    --model neuralmind/bert-base-portuguese-cased \
    --data data/dev_ptbr.json \
    --output predictions/predictions_bertimbau_base.json
```

Exemplo utilizando o NorBERTo:

```bash
python eval_model.py \
    --model Itau-Unibanco/NorBERTo-base \
    --data data/dev_ptbr.json \
    --output predictions/predictions_norberto_base.json
```

Exemplo utilizando a Albertina:

```bash
python eval_model.py \
    --model PORTULAN/albertina-100m-portuguese-ptbr-encoder \
    --data data/dev_ptbr.json \
    --output predictions/predictions_albertina_100m.json
```

Opções úteis:

```bash
# Avalia apenas exemplos intrassentenciais
python eval_model.py --model <model_id> --skip-intersentence

# Executa sem GPU
python eval_model.py --model <model_id> --no-cuda
```

### 3. Calcular as Métricas

Avaliar um único modelo:

```bash
python evaluation.py \
    --data data/dev_ptbr.json \
    --predictions predictions/predictions_bertimbau_base.json
```

Avaliar todos os modelos disponíveis:

```bash
python evaluation.py \
    --data data/dev_ptbr.json \
    --predictions-dir predictions/ \
    --output results.json
```

---

## Métricas

| Métrica        | Descrição                                                                                   | Valor Desejável |
| -------------- | ------------------------------------------------------------------------------------------- | --------------- |
| **LM Score**   | Capacidade do modelo de preferir sentenças semanticamente relacionadas às não relacionadas  | Alto            |
| **SS Score**   | Tendência do modelo em preferir associações estereotipadas em relação às antiestereotipadas | Próximo de 50   |
| **ICAT Score** | Combinação entre qualidade linguística e neutralidade                                       | Alto            |

### Interpretação do SS Score

* **50** → Sem preferência entre estereótipos e antiestereótipos.
* **> 50** → Preferência por associações estereotipadas.
* **< 50** → Preferência por associações antiestereotipadas.

---

## Referência

[StereoSet: Measuring stereotypical bias in pretrained language models](https://aclanthology.org/2021.acl-long.416/) (Nadeem et al., ACL-IJCNLP 2021)
