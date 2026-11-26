"""
Repete o teste "cor prevê fonte" (o que deu 0.8631 antes da correção),
mas no pool de TREINO (dataset_split/augmented/), onde o jitter de matiz
foi de fato aplicado — não no teste, que nunca é tocado por augmentation
e por isso nunca muda, independente de qualquer correção no pipeline.

Se o jitter funcionou, cor prevendo fonte deve cair bastante em relação
ao 0.8631 anterior (medido nas imagens originais, sem jitter).

Uso:
    python 12_checar_atalho_pos_jitter.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))
from config import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

TAMANHO_REDUZIDO = (64, 64)
BINS_HISTOGRAMA = 32


def extrair_features_vegetacao(path):
    img = Image.open(path).convert('RGB').resize(TAMANHO_REDUZIDO)
    arr = np.array(img, dtype=np.float64) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mascara = (g > r) & (g > b)
    if mascara.sum() < 20:
        mascara = np.ones_like(mascara, dtype=bool)

    hist_rgb = []
    for canal_arr in (r, g, b):
        pixels = canal_arr[mascara]
        h, _ = np.histogram(pixels, bins=BINS_HISTOGRAMA, range=(0, 1))
        hist_rgb.append(h / (h.sum() + 1e-8))
    hist_rgb = np.concatenate(hist_rgb)

    stats = np.array([r[mascara].mean(), r[mascara].std(),
                       g[mascara].mean(), g[mascara].std(),
                       b[mascara].mean(), b[mascara].std(),
                       mascara.mean()])

    return np.concatenate([hist_rgb, stats])


def fonte_do_arquivo(nome):
    if 'usuario_' in nome:
        return 'usuario'
    if 'professor_' in nome:
        return 'professor'
    if 'externo_' in nome:
        return 'externo'
    return None


def main():
    print(f"Lendo pool de treino (com jitter) de: {AUGMENTED_DIR}\n")

    X, fontes = [], []
    for cls_dir in AUGMENTED_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
        for arquivo in cls_dir.iterdir():
            fonte = fonte_do_arquivo(arquivo.name)
            if fonte is None:
                continue
            try:
                X.append(extrair_features_vegetacao(arquivo))
                fontes.append(fonte)
            except Exception:
                continue

    X = np.array(X)
    fontes = np.array(fontes)
    valores, contagens = np.unique(fontes, return_counts=True)
    print(f"{len(X)} imagens originais (com jitter) identificadas por fonte: "
          f"{dict(zip(valores, contagens))}\n")

    if len(set(fontes)) < 2:
        print("Menos de 2 fontes encontradas — não dá pra testar.")
        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=300, random_state=SEED)
    scores = cross_val_score(clf, X, fontes, cv=skf, scoring='f1_macro')

    print(f"F1-macro de cor (só vegetação) prevendo FONTE, COM jitter aplicado: "
          f"{scores.mean():.4f} +- {scores.std():.4f}")
    print(f"Mesmo teste, ANTES da correção (sem jitter, medido no dataset anterior): 0.8631")
    print(f"Acaso ({len(set(fontes))} fontes): ~{1/len(set(fontes)):.4f}\n")

    queda = 0.8631 - scores.mean()
    if scores.mean() < 0.5:
        print(f"[RESOLVIDO] Queda de {queda:.4f} — o jitter reduziu drasticamente a "
              f"previsibilidade de fonte por cor. Correção funcionou.")
    elif queda > 0.2:
        print(f"[MELHOROU, NAO RESOLVIDO] Queda de {queda:.4f} — o jitter ajudou, mas "
              f"ainda sobra sinal de fonte capturável por cor. Pode precisar de faixa "
              f"de jitter maior, ou aceitar como limitação parcialmente mitigada.")
    else:
        print(f"[NAO RESOLVIDO] Queda de apenas {queda:.4f} — o jitter não teve o efeito "
              f"esperado. Precisa investigar se o código realmente está sendo executado "
              f"como planejado antes de confiar nos resultados.")


if __name__ == "__main__":
    main()
