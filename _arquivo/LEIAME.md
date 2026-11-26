# _arquivo

Arquivos tirados da raiz do projeto na limpeza de 25/11/2026. Nada foi apagado.

## debug/
Scripts de verificação pontual e as saídas de texto que eles geraram.

| Arquivo | O que era |
|---|---|
| investigar_pipeline.py / investigacao.txt | varredura do pipeline (classes V7_V8, V4_V5 etc.) |
| resumir_auditoria.py / resumo_auditoria.txt | resumo dos JSONs de auditoria |
| verificar_holdout_e_funil.py / verificacao.txt | funil VOC + vazamento no holdout (substituído pelo 05b) |
| buscar_anotacao.py | procura de arquivos de anotação no banco externo |
| checar_fold4_corrompido.py | checagem de imagens corrompidas no fold 4 |
| checar_vazamento_fold.py | vazamento treino/val com hash 32x32 |
| checar_atalho_nitidez.py | teste "nitidez prevê fonte" |
| 11_checar_jitter_aplicado.py | confirma o jitter de matiz no treino |
| 12_checar_atalho_pos_jitter.py | "cor prevê fonte" pós-jitter (corrigido depois pelo 13) |
| contar_datasets.py, comparar_fontes.py, comparar_erros.py | contagens e comparações rápidas |
| significancia.txt | saída parcial do 06b (o log completo está em outputs/results/significancia_log.txt) |

Para rodar algum deles de novo, copie-o de volta para a raiz do projeto:
eles usam caminhos relativos e alguns importam `config.py`.

## obsoletos/
- `09_gerar_relatorio_visual (1).py`: cópia idêntica do `09_gerar_relatorio_visual.py`
- `harmonizar_dataset.py`: versão com 2 fontes, substituída pelo `montar_dataset_final.py` (3 fontes).
  Ele grava em `dataset_unificado/`, então rodá-lo sobrescreveria o dataset atual.
- `__pycache__/`: cache do Python, é recriado sozinho

## documentos/
Versões do artigo (docx/pdf), PDFs do Overleaf, registro_tentativas_e_fracassos.md e o notebook de referência.
