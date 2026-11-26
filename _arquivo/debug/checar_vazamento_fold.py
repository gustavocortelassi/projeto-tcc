"""
Verifica vazamento residual entre treino e validação de um fold — mais
fino que a deduplicação original (hash 16x16, distância <=5), pra pegar
quase-duplicata que passou despercebida (ex.: quadros de rajada/vídeo
próximos, mas não hash-idênticos o suficiente no critério original).

Para cada imagem de validação, procura o vizinho mais próximo no treino
DA MESMA CLASSE (comparação fora da classe não seria vazamento relevante),
usando um hash mais fino (32x32) e reporta a distância mínima — se muitas
imagens de val tiverem vizinho muito próximo no treino, é vazamento real,
não capacidade do modelo.

Uso:
    python checar_vazamento_fold.py --fold 1
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).parent))
from config import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

TAMANHO_HASH = 32  # mais fino que os 16x16 usados na dedup original
LIMIAR_SUSPEITO = 20  # distância Hamming (de 32*32=1024 bits) — ajustar após ver a distribuição


def hash_fino(path):
    img = Image.open(path).convert('L').resize((TAMANHO_HASH, TAMANHO_HASH))
    arr = np.array(img, dtype=np.float64)
    return (arr > arr.mean()).flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, help='Número do fold (1 a N_FOLDS)')
    args = parser.parse_args()
    fold_idx = args.fold - 1

    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)

    fold = kfold['folds'][fold_idx]
    train_paths, train_labels = fold['train_paths'], fold['train_labels']
    val_paths, val_labels = fold['val_paths'], fold['val_labels']

    print(f"Fold {args.fold}: {len(train_paths)} treino, {len(val_paths)} val")
    print("Calculando hashes (pode levar alguns minutos)...")

    train_por_classe = defaultdict(list)
    for p, l in zip(train_paths, train_labels):
        try:
            train_por_classe[l].append((p, hash_fino(p)))
        except Exception:
            continue

    distancias_minimas = []
    suspeitos = []

    for p_val, l_val in zip(val_paths, val_labels):
        try:
            h_val = hash_fino(p_val)
        except Exception:
            continue

        candidatos = train_por_classe.get(l_val, [])
        if not candidatos:
            continue

        dist_min = min(np.count_nonzero(h_val != h_train) for _, h_train in candidatos)
        distancias_minimas.append(dist_min)

        if dist_min <= LIMIAR_SUSPEITO:
            p_train_mais_proximo = min(candidatos, key=lambda c: np.count_nonzero(h_val != c[1]))[0]
            suspeitos.append((p_val, p_train_mais_proximo, dist_min))

    distancias_minimas = np.array(distancias_minimas)
    print(f"\nDistância mínima val->treino (mesma classe), em bits de {TAMANHO_HASH*TAMANHO_HASH}:")
    print(f"  Média: {distancias_minimas.mean():.1f} | Mediana: {np.median(distancias_minimas):.1f} | "
          f"Mínima: {distancias_minimas.min()} | Máxima: {distancias_minimas.max()}")

    print(f"\nImagens de val com vizinho suspeito no treino (distância <= {LIMIAR_SUSPEITO}): "
          f"{len(suspeitos)} de {len(distancias_minimas)} ({100*len(suspeitos)/len(distancias_minimas):.1f}%)")

    if suspeitos:
        print("\nExemplos (val | treino mais próximo | distância):")
        for p_val, p_train, d in suspeitos[:15]:
            print(f"  {Path(p_val).name}  <->  {Path(p_train).name}  (dist={d})")
        if len(suspeitos) > 15:
            print(f"  ... e mais {len(suspeitos) - 15}")

        print(f"\n[ALERTA] {len(suspeitos)} imagens de validação têm vizinho quase-idêntico no "
              f"treino, que a deduplicação original não pegou. Isso explica um val_f1 "
              f"artificialmente alto — precisa expandir o critério de agrupamento do K-Fold "
              f"(hash mais fino, ou threshold mais permissivo) e reprocessar antes de treinar "
              f"os outros modelos.")
    else:
        print("\nNenhum vizinho suspeito encontrado com esse limiar — o val_f1 alto pode ser "
              "legítimo. Considerar que esse dataset é mais fácil que o anterior (mais dado "
              "real, borda já removida, resolução já harmonizada).")


if __name__ == "__main__":
    main()
