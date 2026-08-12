# StereoSet-PTBR

StereoSet-PTBR é uma adaptação do benchmark StereoSet para o português brasileiro, desenvolvida para avaliar viés estereotípico em modelos de linguagem pré-treinados.

Além da tradução do conjunto de dados original, o projeto adapta o protocolo de avaliação para arquiteturas contemporâneas de linguagem utilizadas em português brasileiro, mascaradas e autorregressivas.

O benchmark permite medir associações estereotípicas em diferentes grupos sociais por meio das métricas propostas por Nadeem et al. (2021): **Language Modeling Score (LMS)**, **Stereotype Score (SS)** e **Idealized CAT Score (ICAT)**.

---

## Principais Contribuições

* Tradução do benchmark StereoSet para português brasileiro, com revisão manual de consistência estrutural.
* Reconstrução do dataset preservando a estrutura original de clusters.
* Compatibilidade com arquiteturas de linguagem mascaradas (`AutoModelForMaskedLM`) e autorregressivas (`AutoModelForCausalLM`).
* Pontuação por Pseudo-Log-Likelihood (PLL) com mascaramento por subtoken, alinhado ao tamanho real do termo-alvo em cada tokenizador.
* Pipeline reproduzível para avaliação de viés em modelos de linguagem para português.
* Disponibilização pública do StereoSet-PTBR.

---

## Objetivos

* Adaptar o StereoSet para o português brasileiro.
* Avaliar viés social em modelos de linguagem treinados para português.
* Fornecer um pipeline reproduzível para comparação entre arquiteturas mascaradas e autorregressivas.
* Servir como base para pesquisas em justiça algorítmica, viés e mitigação em modelos de linguagem.

---

## Benchmark

O StereoSet-PTBR preserva a estrutura do benchmark original proposto por Nadeem et al. (2021).

Cada exemplo pertence a um *cluster* composto por:

* Sentença estereotipada;
* Sentença antiestereotipada;
* Sentença não relacionada.

Os exemplos estão organizados nas seguintes categorias de viés:

* Gênero
* Profissão
* Raça
* Religião

### Estatísticas do Dataset

| Tarefa         | Quantidade     |
| -------------- | -------------- |
| Intra-sentence | 2.106 clusters |
| Inter-sentence | 2.123 clusters |

### Controle de Qualidade da Tradução

O conjunto de dados foi traduzido automaticamente (`openai/gpt-4.1-mini`, via OpenRouter, temperatura 0) e revisado manualmente para verificar a integridade estrutural dos exemplos e a correta preservação do marcador `BLANK`. A revisão identificou e corrigiu 517 sentenças (11,1% dos 4.229 clusters) em que o marcador havia sido incompletamente substituído (tarefa *intra-sentence*) ou indevidamente inserido (tarefa *inter-sentence*). `ptbr_llm.csv` já reflete o dataset corrigido.

---

## Diferenças em Relação ao StereoSet Original

O StereoSet original foi desenvolvido exclusivamente para o inglês e utiliza o mecanismo de **Next Sentence Prediction (NSP)** para avaliação da tarefa *inter-sentence*.

Nesta adaptação:

* Os exemplos foram traduzidos para português brasileiro.
* O suporte foi estendido para modelos mascarados e autorregressivos contemporâneos treinados em português.
* Ambas as tarefas (*intra-sentence* e *inter-sentence*) são pontuadas de forma uniforme por **Pseudo-Log-Likelihood (PLL)** para modelos mascarados, em substituição ao NSP.
* Modelos autorregressivos são pontuados via *teacher forcing* em um único *forward pass* por sentença, sem qualquer mecanismo de mascaramento.
* O pipeline foi generalizado para diferentes arquiteturas e tokenizadores (WordPiece, SentencePiece, BPE), com alinhamento por subtoken para lidar corretamente com termos-alvo que se fragmentam em mais de um token.

> Os resultados obtidos neste repositório não devem ser comparados numericamente aos valores reportados no artigo original do StereoSet. As comparações são válidas apenas entre modelos avaliados sob o mesmo protocolo experimental.

---

## Estrutura do Projeto

```text
stereoset-ptbr/
├── dev_ptbr.json
├── predictions_*.json
├── results.json
├── subtoken_distribution.json
├── stereoset-ptbr.ipynb│
├── analyze_subtokens.py
├── convert_csv_to_json.py
├── dataloader.py
├── eval_model.py
├── eval_autoregressive.py
├── evaluation.py
├── ptbr_llm.csv
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Modelos Avaliados

| Modelo          | Hugging Face ID                                   | Arquitetura   | Parâmetros | Licença           |
| --------------- | -------------------------------------------------- | ------------- | ---------- | ----------------- |
| BERTimbau Base  | `neuralmind/bert-base-portuguese-cased`            | BERT          | 110M       | MIT                |
| NorBERTo Base   | `Itau-Unibanco/NorBERTo-base`                      | ModernBERT    | 150M       | CC-BY-NC-SA 4.0    |
| Albertina PT-BR | `PORTULAN/albertina-900m-portuguese-ptbr-encoder`  | DeBERTa-v2    | 900M       | MIT                |
| Tucano-1b1      | `TucanoBR/Tucano-1b1`                              | Decoder-only  | 1,1B       | Apache 2.0         |

Os três primeiros modelos são de linguagem mascarada (MLM) e compartilham o mesmo procedimento de pontuação por PLL. O Tucano-1b1 é autorregressivo (*decoder-only*) e é pontuado por um procedimento próprio via *teacher forcing* (Seção "Pontuação de Modelos Autorregressivos").

O pipeline pode ser utilizado com qualquer modelo compatível com `AutoModelForMaskedLM` (via `eval_model.py`) ou `AutoModelForCausalLM` (via `eval_autoregressive.py`). O sistema detecta automaticamente características arquiteturais, como suporte a `token_type_ids`, permitindo avaliar diferentes famílias de modelos sem alterações no código.

---

## Instalação

Clone o repositório e instale as dependências:

```bash
git clone <repository_url>
cd stereoset-ptbr

pip install -r requirements.txt
```

`bitsandbytes` e `accelerate` são necessários apenas para carregamento quantizado de modelos grandes em `eval_autoregressive.py` (opção `--load-in-8bit`).

---

## Fluxo Experimental

### 1. Converter o Dataset

Converte o dataset traduzido para o formato JSON compatível com o StereoSet.

```bash
python convert_csv_to_json.py \
    --input ptbr_llm.csv \
    --output data/dev_ptbr.json
```

### 2. Executar a Avaliação

**Modelos de linguagem mascarada** (BERTimbau, NorBERTo, Albertina), via PLL:

```bash
python eval_model.py \
    --model neuralmind/bert-base-portuguese-cased \
    --data data/dev_ptbr.json \
    --output predictions/predictions_bertimbau_base.json
```

```bash
python eval_model.py \
    --model Itau-Unibanco/NorBERTo-base \
    --data data/dev_ptbr.json \
    --output predictions/predictions_norberto_base.json
```

```bash
python eval_model.py \
    --model PORTULAN/albertina-900m-portuguese-ptbr-encoder \
    --data data/dev_ptbr.json \
    --output predictions/predictions_albertina_900m.json
```

**Modelo autorregressivo** (Tucano-1b1), via teacher forcing:

```bash
python eval_autoregressive.py \
    --model TucanoBR/Tucano-1b1 \
    --data data/dev_ptbr.json \
    --output predictions/predictions_tucano_1b1.json
```

Opções úteis (ambos os scripts):

```bash
# Avalia apenas exemplos intrassentenciais
python eval_model.py --model <model_id> --skip-intersentence

# Executa sem GPU
python eval_model.py --model <model_id> --no-cuda

# Carregamento quantizado (modelos grandes, só eval_autoregressive.py)
python eval_autoregressive.py --model <model_id> --load-in-8bit
```

---

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
    --output results/results.json
```

---

## Pontuação de Modelos Mascarados via PLL

Cada subtoken da sequência avaliada é mascarado individualmente, mantendo os demais subtokens com seus valores verdadeiros; o escore final corresponde à média das log-probabilidades estimadas para cada subtoken mascarado (Salazar et al., 2020). Na tarefa *inter-sentence*, a sequência avaliada é a sentença candidata inteira, condicionada ao contexto. Na tarefa *intra-sentence*, a sequência avaliada é o termo-alvo, decomposto em seus subtokens: cada subtoken do termo é mascarado e pontuado individualmente, um de cada vez, mantendo os demais subtokens do próprio termo e todo o contexto ao redor com seus valores verdadeiros.

O número de subtokens de cada termo-alvo é determinado por alinhamento de prefixo e sufixo contra a tokenização da sentença completa (ver `analyze_subtokens.py` e `analysis/subtoken_distribution.json`), necessário porque tokenizadores SentencePiece (Albertina) representam um termo de forma diferente conforme o contexto em que aparece. A maioria dos termos-alvo do StereoSet-PTBR fragmenta-se em mais de um subtoken: 64,9% no BERTimbau, 53,6% no NorBERTo e 80,1% na Albertina.

## Pontuação de Modelos Autorregressivos via Teacher Forcing

Modelos autorregressivos (*decoder-only*) dispensam qualquer mecanismo de mascaramento: como a sequência já é processada da esquerda para a direita por construção, a probabilidade de cada subtoken é obtida diretamente da distribuição prevista pelo modelo na posição imediatamente anterior a ele, em um único *forward pass* por exemplo (*teacher forcing*) — o modelo recebe como contexto os tokens verdadeiros da sequência, e não suas próprias previsões anteriores. Essa métrica não é uma PLL, já que não há mascaramento; corresponde à log-likelihood média dos subtokens da sequência avaliada sob teacher forcing.

---

## Métricas

| Métrica  | Descrição                                                                   | Valor Desejável |
| -------- | --------------------------------------------------------------------------- | --------------- |
| **LMS**  | Capacidade do modelo de preferir sentenças relacionadas às não relacionadas | Alto            |
| **SS**   | Tendência do modelo em preferir associações estereotipadas                  | Próximo de 50   |
| **ICAT** | Combinação entre qualidade linguística e neutralidade                       | Alto            |

### Interpretação do SS Score

* **50** → Sem preferência entre estereótipos e antiestereótipos.
* **> 50** → Preferência por associações estereotipadas.
* **< 50** → Preferência por associações antiestereotipadas.

---

## Formulação das Métricas

O StereoSet-PTBR utiliza exatamente as mesmas métricas definidas no benchmark original StereoSet (Nadeem et al., 2021).

Cada *cluster* contém:

* Sentença estereotipada ($s_{pro}$)
* Sentença antiestereotipada ($s_{anti}$)
* Sentença não relacionada ($s_{unr}$)

Após a avaliação pelo modelo, cada sentença recebe um escore $f(s)$.

### Preferência por Estereótipos

$$
pro(t)=
\sum_{\text{clusters de }t}
\mathbf{1}[f(s_{pro}) > f(s_{anti})]
$$

### Preferência por Sentenças Relacionadas

$$
related(t)=
\sum_{\text{clusters de }t}
\left(
\mathbf{1}[f(s_{pro}) > f(s_{unr})]
+
\mathbf{1}[f(s_{anti}) > f(s_{unr})]
\right)
$$

### Language Modeling Score (LMS)

$$
LMS(t)=
\frac{related(t)}
{2 \cdot total(t)}
\times 100
$$

O LMS mede a capacidade do modelo de distinguir sentenças semanticamente relacionadas ao contexto de sentenças não relacionadas.

### Stereotype Score (SS)

$$
SS(t)=
\frac{pro(t)}
{total(t)}
\times 100
$$

O SS mede a tendência do modelo de preferir associações estereotipadas em detrimento das antiestereotipadas.

### Idealized CAT Score (ICAT)

$$
ICAT =
LMS \times
\frac{\min(SS,\ 100-SS)}
{50}
$$

O ICAT combina qualidade linguística e neutralidade em relação aos estereótipos.

---

## Reprodutibilidade

Todos os experimentos apresentados podem ser reproduzidos utilizando o notebook:

```text
notebooks/stereoset-ptbr.ipynb
```

O notebook executa:

1. Conversão do dataset;
2. Avaliação dos modelos mascarados e do modelo autorregressivo;
3. Cálculo das métricas;
4. Consolidação dos resultados em `results/results.json`.

---

## Licença

Este projeto — código, dataset traduzido e resultados — está licenciado sob **CC BY-SA 4.0** (Creative Commons Atribuição-CompartilhaIgual 4.0 Internacional). Veja [LICENSE](LICENSE).

Os modelos avaliados mantêm suas licenças originais (ver tabela em "Modelos Avaliados"); este repositório não redistribui pesos de modelos.

---

## Referência

NADEEM, M.; BETHKE, A.; REDDY, S. StereoSet: Measuring stereotypical bias in pretrained language models. In: *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing* (Volume 1: Long Papers). 2021. p. 5356–5371.

SALAZAR, J. et al. Masked language model scoring. In: *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. 2020. p. 2699–2712.
