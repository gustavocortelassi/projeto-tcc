# Registro de tentativas, cenários e fracassos — TCC Fenologia do Milho

Documento de referência com tudo que foi tentado, o que deu errado, e como foi resolvido, do diagnóstico inicial até o estado atual. Organizado por fase, em ordem cronológica.

---

## Fase 1 — Diagnóstico inicial e reconstrução do pipeline

**Estado encontrado:** pipeline em notebooks quebrados, treino nunca passou da época 2 em nenhum dos 5 modelos. `config.py` desatualizado em relação ao `dataset_raw` real (8 classes no código, 14 pastas no disco). Faltava `batch_size` por modelo, `StratifiedGroupKFold` nunca implementado (risco de vazamento entre cópias aumentadas da mesma foto), `03_train_resnet50` nunca existiu.

**Decisão:** reconstruir do zero como scripts `.py` numerados, não notebooks — elimina ambiguidade de estado de kernel e permite rodar sem supervisão.

---

## Fase 2 — Inventário e decisões de balanceamento

**01_eda.py** revelou contagem real por classe. **V8** (36 imagens, segmentação manual — método de captura diferente do resto) excluído por justificativa dupla: abaixo do limiar mínimo e confundidor de enquadramento.

**Primeira tentativa de `TARGET_N_AUG`:** 1500, emprestado sem crítica de outro contexto. Usuário questionou ("não é só pq eu disse 500 que deve ser 500"), obrigando a reconsiderar duas vezes:
1. Primeira revisão: piso sem teto (preserva dado real das classes abundantes).
2. Segunda revisão, após perceber o confundidor: **H1 do TCC compara CNN x Transformer especificamente em estádios vegetativos** — que são as classes mais escassas. Volume de treino desigual entre vegetativo/reprodutivo confundiria qualquer resultado. Voltou a alvo simétrico (500), agora justificado por multiplicador de augmentation (3,3× máximo para a classe mais escassa), não por conveniência.

---

## Fase 3 — Ambiente: três categorias de erro que se repetiram

- **`py` vs `python`**: o launcher do Windows ignora ambiente conda ativo. Causou execução em Python sem CUDA múltiplas vezes.
- **Dois ambientes conda (`tcc` e `tcc_milho`)** com pacotes diferentes — `tcc_milho` faltava `torchmetrics` na primeira checagem.
- **PowerShell vs cmd.exe**: comandos de um não funcionam no outro (`Get-Content`, `Get-Item`, `.ps1` abrindo no Bloco de Notas em vez de executar). Aconteceu repetidamente ao longo de todo o projeto.
- **Diretório errado**: comandos rodados fora da pasta do projeto, múltiplas vezes.

**Benchmark de throughput (`00_benchmark_treino.py`)** inicialmente reportou `Device: cpu` — diagnosticado como causa do `py` pegando Python sem CUDA. Corrigido, benchmark real confirmou 2 meses de prazo era suficiente mesmo no cenário mais caro (~63h de GPU para os 5 modelos).

---

## Fase 4 — Primeiro treinamento completo: três bugs de execução

1. **`LR_HEAD` não definida** — `config.py` local desatualizado (arquivo não substituído após edição), mesmo erro nos 5 modelos.
2. **Arquivo "sumindo" durante o treino** (`Cópia de IMG-...jpg`) — diagnosticado errado a princípio como antivírus/quarentena. Causa real: `kfold_config.json` lido sem `encoding='utf-8'`, corrompendo nomes de arquivo com acento (`ó` → `Ã³`). Corrigido, e adicionada resiliência no `Dataset` (arquivo ilegível não derruba o processo, usa outra imagem no lugar).
3. **Falso alarme do Swin "travado"** — após 12h sem atualizar o log, presumi estouro de VRAM e recomendei matar o processo. Causa real: buffer de saída do Python não sendo escrito no arquivo de log (resolvido depois com `python -u`). O treino estava funcionando o tempo todo — fold 1 já tinha terminado com val F1 0,90 / test F1 0,92. **Erro reconhecido explicitamente**: recomendação de interromper um processo de 12h baseada em uma única evidência ambígua, sem checar a explicação alternativa mais óbvia (fold já concluído) antes de tratar a hipótese mais dramática como fato.

---

## Fase 5 — Resultados, significância e o padrão do Swin

Treino completo dos 5 modelos: ranking inicial EfficientNetV2 > MobileViT > ResNet50 > ViT > Swin, todos F1 > 0,97 — nível suspeito de homogêneo (motivo detalhado na Fase 7).

**06_teste_significancia.py**: Friedman confirma diferença real entre arquiteturas (p=0,0043). Wilcoxon pareado mostra as 4 comparações envolvendo Swin no teto estrutural de p≈0,0625 (o máximo possível com N=5 folds) — consistência direcional forte, mesmo sem cruzar p<0,05 corrigido.

**05_avaliar_final.py** (F1 por classe): Swin especificamente fraco no par **V6→V7** (estádios vegetativos adjacentes) — achado central que sustenta H1.

---

## Fase 6 — Interpretabilidade (Grad-CAM / atenção)

**07_interpretabilidade.py**: Grad-CAM funcionou de primeira para ResNet50/EfficientNetV2 (API `forward_features`/`forward_head` do timm). Attention Rollout **quebrou no Swin** — erro de tensores incompatíveis (16 vs 64), causa: Swin é hierárquico com atenção em janela, camadas de estágios diferentes operam em resoluções incompatíveis, rollout acumulado não é válido. Corrigido usando só a atenção da última camada (mesmo critério do Grad-CAM), aplicado uniformemente a ViT e Swin.

Inspeção visual (9 imagens): EfficientNetV2 focado e pontual na região de crescimento; ResNet50 concentrado mas espalhado pela planta; **Swin disperso em estrias por múltiplas plantas simultaneamente**; ViT pontos discretos e pequenos. Padrão visual consistente com H1.

---

## Fase 7 — A maior descoberta: atalho de borda preta (letterbox)

Pergunta do usuário ("como validar se não é alucinação") levou à auditoria de shortcut learning.

**08_checar_atalho_bordas.py**: primeira rodada (teste de 13 classes) concluiu "não é atalho" — **conclusão errada**, escondendo uma separação binária clara nas médias por classe. Corrigido adicionando teste de grupo (com borda / sem borda): **F1 = 1,0000, separação perfeita**. Confirmado: o `transform` (`Resize` simples) não removia a borda, os modelos literalmente viam o pixel preto.

**09_avaliar_sem_borda.py** (teste nos modelos já treinados, sem retreinar): queda média de F1 de +0,23 ao remover a borda do teste, com **ViT (-0,47) e Swin (-0,46) colapsando**. Ambíguo entre duas causas: dependência real do atalho vs. sensibilidade de Transformers a mudança de distribuição (literatura já documenta isso). Não dava pra separar as duas sem retreinar.

**Correção completa**: `10_remover_bordas_raw.py` gerou espelho `dataset_raw_sem_borda` (13.273 imagens, 5.691 com borda cortada — 43%). Reprocessamento e **retreino completo #2**.

**Resultado pós-correção**: quedas pequenas (a maioria subiu, na verdade) — confirmou que o colapso do teste anterior era majoritariamente descompasso de distribuição, não dependência pesada do atalho. **O achado do Swin em V6/V7 sobreviveu ao retreino independente** — mesmo padrão de p=0,0625 nas 4 comparações envolvendo Swin, mais uma segunda linha de evidência nova: Swin tem variância entre folds ~10x maior que os outros modelos.

---

## Fase 8 — Erros de execução do ciclo de correção da borda

- `RAW_DIR` apontado pra pasta errada (`dataset_raw_sem_borda` vazia) ao rodar o script que deveria gerá-la — corrigido revertendo temporariamente, rodando, revertendo de novo.
- `treinar_todos.bat` reordenado a pedido do usuário (Swin por último) sem impacto na validade — só sequenciamento.
- Dúvida legítima se o retreino rodou de verdade ou pulou tudo (arquivos de resultado antigos não apagados antes do `.bat`) — confirmado via inspeção do log completo (curva de aprendizado real presente).
- Temporal chegando durante o treino — `Ctrl+C` seguro recomendado proativamente, checkpoint por época garante perda mínima.

---

## Fase 9 — Investigação de duplicata (V1, V3, V4)

Suspeita antiga (V1 e V3 batendo exatamente em 2500 imagens cada) finalmente investigada com `11_checar_duplicatas.py`.

**Resultado: muito pior que a suspeita original.** V1: 41,6% duplicata exata. V3: 45%. **V4** (nem estava sob suspeita): **66%** — pior que os dois. Quase-duplicatas (hash perceptual) ainda mais extensas (1.971 pares em V1, 5.491 em V3). Outras 10 classes: quase zero.

**Causa raiz**: nome de arquivo (`train_set__IMG_1822_01__obj05_ZEAMX_V1.jpg`) revelou origem de um dataset externo de detecção de objetos, com split treino/validação próprio, mesclado sem deduplicar. Confirmado pelo usuário: banco de ~15.000 imagens cedido pelo orientador, cobrindo V1/V3/V4 — bate exatamente com "Fonte C" no `CLASS_ORIGEM` já documentado.

**Gravidade real**: duplicata com nome de arquivo diferente **não era pega pelo agrupamento de K-Fold** (que agrupava por nome de arquivo) — vazamento genuíno entre treino/validação/teste, não só inflação de contagem.

**Correção**: reescrita de `02_preprocessing.py` com deduplicação por hash de conteúdo (MD5 exato + hash perceptual via Union-Find), usando o cluster de conteúdo como chave de agrupamento do K-Fold em vez do nome de arquivo. Escrito, mas **retreino em cima dessa correção específica não chegou a rodar** — o projeto pivotou para reinício com nova estrutura de dataset antes disso.

---

## Fase 10 — Reinício: "Projeto Final" com fontes validadas incrementalmente

Decisão do usuário: recomeçar do zero, adicionando fontes de dado uma de cada vez, validando cada uma antes de mesclar — em vez de descobrir problema por auditoria forense depois do fato consumado.

**Novas fontes**: Dataset 1 (fotos próprias, S24 Ultra, alta resolução) + Dataset 2 (do orientador, já em 224×224).

**`comparar_fontes.py`** revelou taxonomia de classe incompatível entre as fontes: Dataset 1 usava esquema antigo fundido (`V4_V5`, `V7_V8`), Dataset 2 já usava separado (`V4`, `V7`). Resolvido por decisão do usuário: fundir `V4_V5→V4`, `V7_V8→V7`.

**Notebook de um projeto irmão do orientador (soja)** revelou o transform real usado por ele (`Resize((256,256))` + `CenterCrop(224)`) — usado como referência de harmonização, em vez de suposição.

**`harmonizar_dataset.py`**: combina remoção automática de borda (detectada por imagem, não presumida por fonte) + harmonização de resolução idêntica ao pipeline do orientador + fusão de classe. Resultado: **confirmação absoluta** — 100% das imagens do Dataset 2 tinham borda, 0% do Dataset 1.

---

## Fase 11 — Atalho residual de nitidez, e um erro de previsão corrigido

Após remover a borda, hipótese de um atalho secundário: imagens do orientador (originalmente pequenas, cortadas, reampliadas) mais borradas que as do usuário (alta resolução nativa).

**`checar_atalho_nitidez.py`**: teste agregado de 6 classes "puras" deu F1=0,32 (ambíguo). Adicionado teste binário específico (R3_R4, única classe pura do usuário, contra as 5 puras do orientador) — **previsão errada de que sairia >0,80** baseada só na diferença de médias, sem pesar o desvio-padrão largo o suficiente para gerar sobreposição. Resultado real: **F1=0,58, perto do acaso (0,50)**. Erro de previsão reconhecido explicitamente. Conclusão: atalho residual fraco, não vale o custo de degradar imagens de propósito — documentar como limitação é suficiente.

---

## Estado atual

Dataset unificado (`dataset_unificado/`) com borda removida, resolução harmonizada, taxonomia consolidada em 13 classes. Ainda pendente: rodar balanceamento (`TARGET_N_AUG`) e K-Fold com deduplicação por hash (a correção da Fase 9 precisa ser adaptada pra essa pasta nova) antes do próximo retreino completo.
