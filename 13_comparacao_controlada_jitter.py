"""
Comparação controlada: as MESMAS imagens de teste, com e sem jitter de
matiz aplicado on-the-fly (só para este diagnóstico, nada é salvo em
disco nem usado para avaliação real) — isola o efeito do jitter sem
confundir com diferença de amostra (o erro do script anterior, que
comparava teste sem jitter contra treino com jitter, duas amostras
diferentes).

Uso:
    python 13_comparacao_controlada_jitter.py
"""

import sys
import json
import random
from pathlib import Path

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


def variar_matiz(img, graus):
    deslocamento = int(round(graus / 360 * 256))
    hsv = img.convert('HSV')
    h, s, v = hsv.split()
    h_arr = (np.array(h, dtype=np.int16) + deslocamento) % 256
    h_novo = Image.fromarray(h_arr.astype(np.uint8), mode='L')
    return Image.merge('HSV', (h_novo, s, v)).convert('RGB')


def variar_hsv_completo(img, graus_h, fator_s, fator_v):
    """Varia os três canais — hipótese: a marca de fonte pode estar mais em
    saturação/brilho do que em matiz, que foi o único canal testado antes."""
    deslocamento = int(round(graus_h / 360 * 256))
    hsv = img.convert('HSV')
    h, s, v = hsv.split()
    h_arr = (np.array(h, dtype=np.int16) + deslocamento) % 256
    s_arr = np.clip(np.array(s, dtype=np.float64) * fator_s, 0, 255)
    v_arr = np.clip(np.array(v, dtype=np.float64) * fator_v, 0, 255)
    h_novo = Image.fromarray(h_arr.astype(np.uint8), mode='L')
    s_novo = Image.fromarray(s_arr.astype(np.uint8), mode='L')
    v_novo = Image.fromarray(v_arr.astype(np.uint8), mode='L')
    return Image.merge('HSV', (h_novo, s_novo, v_novo)).convert('RGB')


def extrair_features_vegetacao(img):
    img = img.resize(TAMANHO_REDUZIDO)
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


def fonte_do_path(path_str):
    stem = Path(path_str).stem
    if 'usuario_' in stem:
        return 'usuario'
    if 'professor_' in stem:
        return 'professor'
    if 'externo_' in stem:
        return 'externo'
    return None


def main():
    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)

    print(f"Lendo {len(kfold['test_paths'])} imagens de teste...\n")

    X_sem_jitter, X_com_jitter_h, X_com_jitter_hsv, X_com_jitter_agressivo, fontes = [], [], [], [], []
    rng = random.Random(SEED)

    for path in kfold['test_paths']:
        fonte = fonte_do_path(path)
        if fonte is None:
            continue
        try:
            img_original = Image.open(path).convert('RGB')
            X_sem_jitter.append(extrair_features_vegetacao(img_original))

            graus = rng.uniform(-18, 18)
            img_jitter_h = variar_matiz(img_original, graus)
            X_com_jitter_h.append(extrair_features_vegetacao(img_jitter_h))

            fator_s = rng.uniform(0.6, 1.4)
            fator_v = rng.uniform(0.6, 1.4)
            img_jitter_hsv = variar_hsv_completo(img_original, graus, fator_s, fator_v)
            X_com_jitter_hsv.append(extrair_features_vegetacao(img_jitter_hsv))

            graus_agr = rng.uniform(-30, 30)
            fator_s_agr = rng.uniform(0.4, 1.8)
            fator_v_agr = rng.uniform(0.5, 1.6)
            img_agressivo = variar_hsv_completo(img_original, graus_agr, fator_s_agr, fator_v_agr)
            X_com_jitter_agressivo.append(extrair_features_vegetacao(img_agressivo))

            fontes.append(fonte)
        except Exception:
            continue

    X_sem_jitter = np.array(X_sem_jitter)
    X_com_jitter_h = np.array(X_com_jitter_h)
    X_com_jitter_hsv = np.array(X_com_jitter_hsv)
    X_com_jitter_agressivo = np.array(X_com_jitter_agressivo)
    fontes = np.array(fontes)

    valores, contagens = np.unique(fontes, return_counts=True)
    print(f"{len(fontes)} imagens identificadas por fonte: {dict(zip(valores, contagens))}\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=300, random_state=SEED)

    scores_sem = cross_val_score(clf, X_sem_jitter, fontes, cv=skf, scoring='f1_macro')
    scores_h = cross_val_score(clf, X_com_jitter_h, fontes, cv=skf, scoring='f1_macro')
    scores_hsv = cross_val_score(clf, X_com_jitter_hsv, fontes, cv=skf, scoring='f1_macro')
    scores_agr = cross_val_score(clf, X_com_jitter_agressivo, fontes, cv=skf, scoring='f1_macro')

    print(f"MESMAS {len(fontes)} imagens em todos os casos — só difere o jitter:\n")
    print(f"F1-macro prevendo FONTE, sem jitter:                  {scores_sem.mean():.4f} +- {scores_sem.std():.4f}")
    print(f"F1-macro prevendo FONTE, só matiz (H, +-18):          {scores_h.mean():.4f} +- {scores_h.std():.4f}")
    print(f"F1-macro prevendo FONTE, HSV (H+-18, S/V 0.6-1.4):    {scores_hsv.mean():.4f} +- {scores_hsv.std():.4f}")
    print(f"F1-macro prevendo FONTE, HSV agressivo (H+-30, S 0.4-1.8, V 0.5-1.6): "
          f"{scores_agr.mean():.4f} +- {scores_agr.std():.4f}")

    print(f"\nAcaso (3 fontes): ~0.3333")
    print(f"Queda HSV moderado: {scores_sem.mean() - scores_hsv.mean():.4f}")
    print(f"Queda HSV agressivo: {scores_sem.mean() - scores_agr.mean():.4f}")

    if scores_agr.mean() < 0.5:
        print("\n[RECOMENDAÇÃO] Faixa agressiva chega bem mais perto do acaso — vale usar "
              "essa faixa no pipeline real, não a moderada, antes de retreinar.")
    else:
        print("\n[AVISO] Mesmo agressivo, ainda sobra sinal considerável acima do acaso. "
              "Pode não ser só HSV — considerar investigar textura ou aceitar isso como "
              "limitação residual documentada, dado o tempo que resta.")


if __name__ == "__main__":
    main()
