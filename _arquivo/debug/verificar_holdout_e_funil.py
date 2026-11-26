"""
Verificação SOMENTE LEITURA (não altera, não move, não apaga nada).

1. Funil da extração Pascal VOC: descartes por motivo (objetos_filtrados.json)
2. Composição por fonte (usuario/professor/externo) em unificado, treino e teste
3. Vazamento no holdout: imagens de teste com quase-duplicata no treino+val,
   usando o MESMO critério do pipeline (average hash 16x16, Hamming <= 5)

Uso:
    python verificar_holdout_e_funil.py > verificacao.txt
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, SPLIT_DIR, AUGMENTED_DIR, CLASSES_TREINO

ImageFile.LOAD_TRUNCATED_IMAGES = True
LIMIAR_PIPELINE = 5    # mesmo valor do 02_preprocessing.py
LIMIAR_FOLGADO = 10    # sensibilidade extra, só para referência
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
FONTES = ('usuario', 'professor', 'externo')


def secao(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def fonte_de(nome):
    for f in FONTES:
        if nome.startswith(f + "_"):
            return f
    return "outra"


def ahash(path):
    img = Image.open(path).convert('L').resize((16, 16))
    arr = np.array(img, dtype=np.float64)
    return (arr > arr.mean()).flatten()


# ─── 1. Funil da extração ────────────────────────────────────────────────────
def funil_extracao():
    secao("1. DESCARTES NA EXTRAÇÃO PASCAL VOC (objetos_filtrados.json)")
    arq = Path("objetos_filtrados.json")
    if not arq.exists():
        print("  (arquivo não encontrado)")
        return
    with open(arq, encoding="utf-8") as f:
        dados = json.load(f)

    geral = Counter(d.get('motivo', '?').split(' (')[0] for d in dados)
    print(f"  Total descartado: {len(dados)}")
    for m, n in geral.most_common():
        print(f"    {m:40} {n:>7}")

    print("\n  Milho nos estádios-alvo, por motivo:")
    alvo = defaultdict(Counter)
    for d in dados:
        if d.get('classe') in {'V1', 'V3', 'V4'}:
            alvo[d['classe']][d.get('motivo', '?').split(' (')[0]] += 1
    for cls in ('V1', 'V3', 'V4'):
        for m, n in alvo[cls].most_common():
            print(f"    {cls:4} {m:36} {n:>7}")


# ─── 2. Composição por fonte ─────────────────────────────────────────────────
def originais_trainval(cls):
    """Originais do treino+val = arquivos em augmented/ que não começam com aug_.
    Retorna o caminho equivalente em dataset_unificado (versão sem jitter)."""
    pasta = AUGMENTED_DIR / cls
    if not pasta.exists():
        return []
    return [RAW_DIR / cls / f.name for f in pasta.iterdir()
            if f.is_file() and f.suffix.lower() in EXT and not f.name.startswith("aug_")]


def arquivos_teste(cls):
    pasta = SPLIT_DIR / "test" / cls
    if not pasta.exists():
        return []
    return [f for f in pasta.iterdir() if f.is_file() and f.suffix.lower() in EXT]


def composicao():
    secao("2. COMPOSIÇÃO POR FONTE (usuario / professor / externo)")
    print(f"  {'Classe':8} {'Etapa':10} " + " ".join(f"{f:>10}" for f in FONTES))
    totais = defaultdict(Counter)
    for cls in CLASSES_TREINO:
        etapas = {
            'unificado': [f for f in (RAW_DIR / cls).iterdir()
                          if f.is_file() and f.suffix.lower() in EXT] if (RAW_DIR / cls).exists() else [],
            'treino': originais_trainval(cls),
            'teste': arquivos_teste(cls),
        }
        for etapa, arqs in etapas.items():
            c = Counter(fonte_de(a.name) for a in arqs)
            totais[etapa].update(c)
            print(f"  {cls:8} {etapa:10} " + " ".join(f"{c.get(f, 0):>10}" for f in FONTES))
    print()
    for etapa, c in totais.items():
        print(f"  {'TOTAL':8} {etapa:10} " + " ".join(f"{c.get(f, 0):>10}" for f in FONTES))


# ─── 3. Vazamento no holdout ─────────────────────────────────────────────────
def vazamento_holdout():
    secao(f"3. QUASE-DUPLICATAS ENTRE TESTE E TREINO+VAL (aHash 16x16)")
    print(f"  {'Classe':8} {'teste':>6} {'treino':>7} "
          f"{'<=' + str(LIMIAR_PIPELINE):>6} {'<=' + str(LIMIAR_FOLGADO):>6} {'dist.mín.mediana':>18}")

    exemplos = []
    tot_teste = tot_v5 = tot_v10 = 0
    for cls in CLASSES_TREINO:
        teste = arquivos_teste(cls)
        treino = [p for p in originais_trainval(cls) if p.exists()]
        if not teste or not treino:
            print(f"  {cls:8} (sem dados)")
            continue

        H_te = np.array([ahash(p) for p in teste])
        H_tr = np.array([ahash(p) for p in treino])
        dist = (H_te[:, None, :] != H_tr[None, :, :]).sum(axis=2)
        dmin = dist.min(axis=1)
        idx_viz = dist.argmin(axis=1)

        n5 = int((dmin <= LIMIAR_PIPELINE).sum())
        n10 = int((dmin <= LIMIAR_FOLGADO).sum())
        tot_teste += len(teste); tot_v5 += n5; tot_v10 += n10
        print(f"  {cls:8} {len(teste):>6} {len(treino):>7} {n5:>6} {n10:>6} {np.median(dmin):>18.0f}")

        for i in np.argsort(dmin)[:3]:
            if dmin[i] <= LIMIAR_FOLGADO:
                exemplos.append((cls, int(dmin[i]), teste[i].name, treino[idx_viz[i]].name))

    print(f"\n  TOTAL: {tot_v5} de {tot_teste} imagens de teste com vizinho <= {LIMIAR_PIPELINE} "
          f"({100 * tot_v5 / max(tot_teste, 1):.1f}%) | {tot_v10} com vizinho <= {LIMIAR_FOLGADO} "
          f"({100 * tot_v10 / max(tot_teste, 1):.1f}%)")

    if exemplos:
        print("\n  Exemplos de pares mais próximos (para conferir visualmente):")
        for cls, d, te, tr in exemplos[:20]:
            print(f"    {cls:6} dist={d:<3} teste: {te}  |  treino: {tr}")


def main():
    funil_extracao()
    composicao()
    vazamento_holdout()


if __name__ == "__main__":
    main()
