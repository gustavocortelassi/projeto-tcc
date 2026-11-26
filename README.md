# Classificação de estádios fenológicos do milho com redes profundas

Trabalho de conclusão de curso. Compara cinco arquiteturas de visão computacional
(CNNs, Transformers e um híbrido) na classificação de estádios fenológicos do milho
pela escala de Ritchie, a partir de imagens de campo.

**Classes (13):** V1, V2, V3, V4, V6, V7, V12, VT, R1, R2, R3_R4, R5, R6

## Resultados

Ensemble dos 5 folds (média das probabilidades softmax), avaliado no conjunto de
teste fixo de 761 imagens, separado antes de qualquer augmentation.

| Modelo | Família | F1-macro | Acurácia | Erros | Parâmetros | Latência* |
|---|---|---|---|---|---|---|
| MobileViT v2 | Híbrido | 0,9845 | 0,9882 | 9 | 9,8 M | 10,2 ms |
| EfficientNetV2-S | CNN | 0,9845 | 0,9882 | 9 | 20,2 M | 20,3 ms |
| ResNet50 | CNN | 0,9773 | 0,9842 | 12 | 23,5 M | 7,4 ms |
| ViT-B/16 | Transformer | 0,9761 | 0,9816 | 14 | 85,8 M | 6,7 ms |
| Swin-B | Transformer | 0,9507 | 0,9606 | 30 | 86,9 M | 23,9 ms |

\* batch 1, RTX 3070 Ti, média de 50 medições.

**Estabilidade entre folds** (F1-macro de cada modelo individual no teste):

| Modelo | Média | Desvio padrão |
|---|---|---|
| MobileViT v2 | 0,9841 | 0,0021 |
| EfficientNetV2-S | 0,9826 | 0,0048 |
| ResNet50 | 0,9776 | 0,0050 |
| ViT-B/16 | 0,9583 | 0,0184 |
| Swin-B | 0,9045 | 0,0281 |

**Testes estatísticos**
- Friedman sobre o F1 por fold: χ² = 16,8, p = 0,0021, W de Kendall = 0,84.
- McNemar exato entre pares, com correção de Holm: só os pares que envolvem o Swin
  diferem significativamente. Entre as outras quatro arquiteturas, nenhuma diferença
  é significativa.

**Controles**
- *Vazamento por quase-duplicata:* retirando do teste as imagens com vizinho
  parecido no treino (761 → 699 → 649 imagens), o F1 praticamente não muda
  (MobileViT: 0,9845 → 0,9844 → 0,9840).
- *Dependência de cor:* com perturbação forte de matiz, saturação e brilho, o F1 cai
  pouco (MobileViT 0,9876, EfficientNetV2 0,9828, ResNet50 0,9702). Em tons de cinza
  a queda é grande, sobretudo nos Transformers (ViT 0,11, Swin 0,09), o que mostra
  que a cor é uma informação usada pelos modelos.
- *Acerto por fonte:* nas classes com imagens de mais de uma fonte, o acerto fica
  alto em todas elas (a menor taxa é em V1 do banco do orientador, com 8 imagens),
  o que indica que os modelos não estão apenas reconhecendo a origem da foto.

Os números completos estão em [`outputs/results/`](outputs/results/), em especial
`comparativo_final.csv`, `significancia_log.txt`, `robustez_cor_log.txt` e
`sensibilidade_comparativo.csv`.

## Dados

O dataset une três fontes:

1. **Fotos próprias** (smartphone)
2. **Banco do orientador** (imagens 224×224)
3. **Banco externo** com anotações Pascal VOC, do qual só as classes V1, V3 e V4 de milho são usadas

Todas as imagens passam pelo mesmo tratamento: corte de borda preta, `Resize(256, 256)`
e `CenterCrop(224)`. O banco externo também é deduplicado (MD5 exato), e
quase-duplicatas (average hash 16×16, Hamming ≤ 5) são agrupadas para nunca ficarem
separadas entre treino, validação e teste.

**As imagens e os pesos treinados não estão no repositório** por causa do tamanho
(mais de 5 GB de imagens e 4,3 GB de modelos).

## Pipeline

| Etapa | Script | O que faz |
|---|---|---|
| Montagem | `extrair_bboxes_voc.py` | recorta os objetos anotados do banco externo, por classe |
| | `organizar_fonte_externa.py` | alternativa: organiza o banco externo pelo sufixo do nome do arquivo |
| | `montar_dataset_final.py` | junta as 3 fontes em `dataset_unificado/` e gera `grupos_quase_duplicata.json` |
| Pré-processamento | `02_preprocessing.py` | undersampling, holdout de teste (15%), augmentation e `StratifiedGroupKFold` (5 folds) |
| Treino | `treinar_todos.ps1` ou `03_treinar.py --modelo <nome>` | treina os 5 modelos em K-Fold, com checkpoint e retomada automática |
| Avaliação | `05_avaliar_final.py` | ensemble dos folds no teste, F1 por classe e matriz de confusão |
| | `05b_avaliar_sensibilidade.py` | sensibilidade do teste a quase-duplicatas |
| Estatística | `06_teste_significancia.py`, `06b_significancia_final.py` | Friedman, McNemar, eficiência e estabilidade por classe |
| Interpretabilidade | `07b_gradcam_v6_geral.py`, `08_comparar_folds_atencao.py`, `14_gradcam_acertos.py` | Grad-CAM e mapas de atenção |
| Relatório | `09_gerar_relatorio_visual.py`, `montar_pasta_reuniao.ps1` | gráficos comparativos e por modelo |
| Controles | `10_baseline_trivial.py`, `13_comparacao_controlada_jitter.py`, `15_teste_robustez_cor.py` | baseline sem rede neural e testes de atalho de cor |

A configuração compartilhada (classes, hiperparâmetros, modelos) fica em
[`config.py`](config.py), e a lógica de treino em [`treino_core.py`](treino_core.py).

### Treino

- Modelos pré-treinados no ImageNet (via `timm`).
- 3 épocas com o backbone congelado (lr 1e-3), depois fine-tuning completo (lr 5e-5).
- Até 50 épocas, com early stopping (paciência 10) pelo F1-macro de validação.

## Como rodar

Requer Python 3.11+ e GPU com CUDA.

```bash
pip install torch torchvision timm torchmetrics scikit-learn scipy pandas numpy matplotlib pillow
```

1. Ajuste `PROJECT_ROOT` em [`config.py`](config.py) para o caminho local do projeto.
2. Coloque as três fontes de imagens nas pastas esperadas por `montar_dataset_final.py`.
3. Rode as etapas na ordem da tabela acima, sempre a partir da raiz do projeto:

```bash
python montar_dataset_final.py
python 02_preprocessing.py
powershell -ExecutionPolicy Bypass -File .\treinar_todos.ps1
python 05_avaliar_final.py
python -u 06b_significancia_final.py
```

## Estrutura

```
├── config.py, treino_core.py   configuração e lógica de treino
├── 02_ … 15_*.py               etapas do pipeline
├── treinar_*.py / .ps1 / .bat  treino por modelo ou de todos em sequência
├── outputs/results/            métricas, predições e logs de análise
├── logs/                       logs de treino
└── _arquivo/                   scripts de verificação, versões antigas e documentos do artigo
```
