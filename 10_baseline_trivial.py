"""
Baseline trivial: classificador clássico (histograma de cor RGB/HSV +
estatísticas básicas de textura), sem nenhuma rede neural, no conjunto de
teste fixo.

Não testa um atalho específico (isso já foi feito em borda, duplicata e
nitidez) — testa se a TAREFA INTEIRA é fácil demais para esse dataset,
resolvível por um método raso. Se o baseline chegar perto do F1 dos
modelos profundos (~0,97-0,99), o resultado alto reflete uma tarefa fácil
nesse dataset específico, não necessariamente capacidade real dos
modelos. Se ficar bem abaixo, é evidência a favor de que os modelos estão
aprendendo morfologia de verdade, não um sinal raso qualquer.

Uso:
    python 10_baseline_trivial.py
"""

import sys
import json
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


def extrair_features(path):
    img = Image.open(path).convert('RGB').resize(TAMANHO_REDUZIDO)
    arr = np.array(img, dtype=np.float64) / 255.0

    hist_rgb = []
    for canal in range(3):
        h, _ = np.histogram(arr[:, :, canal], bins=BINS_HISTOGRAMA, range=(0, 1))
        hist_rgb.append(h / h.sum())
    hist_rgb = np.concatenate(hist_rgb)

    img_hsv = np.array(Image.open(path).convert('HSV').resize(TAMANHO_REDUZIDO), dtype=np.float64) / 255.0
    hist_hsv = []
    for canal in range(3):
        h, _ = np.histogram(img_hsv[:, :, canal], bins=BINS_HISTOGRAMA, range=(0, 1))
        hist_hsv.append(h / h.sum())
    hist_hsv = np.concatenate(hist_hsv)

    stats = np.array([
        arr[:, :, 0].mean(), arr[:, :, 0].std(),
        arr[:, :, 1].mean(), arr[:, :, 1].std(),
        arr[:, :, 2].mean(), arr[:, :, 2].std(),
        arr.mean(),
    ])

    return np.concatenate([hist_rgb, hist_hsv, stats])


def extrair_features_vegetacao(path):
    """Mesmo histograma, mas só nos pixels que parecem vegetação (verde
    dominante) — isola cor da planta, ignora solo/fundo. Máscara crua
    (excess green: G maior que R e B), não é segmentação de verdade, mas
    é suficiente pra separar 'cor de planta' de 'cor de cena inteira'."""
    img = Image.open(path).convert('RGB').resize(TAMANHO_REDUZIDO)
    arr = np.array(img, dtype=np.float64) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mascara = (g > r) & (g > b)

    if mascara.sum() < 20:  # poucos pixels de "verde" — imagem sem vegetação clara
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
                       mascara.mean()])  # fração de "verde" na imagem, de bônus

    return np.concatenate([hist_rgb, stats])


def main():
    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)

    print(f"Extraindo features clássicas de {len(kfold['test_paths'])} imagens de teste...")
    X, y = [], []
    for path, label in zip(kfold['test_paths'], kfold['test_labels']):
        try:
            X.append(extrair_features(path))
            y.append(label)
        except Exception as e:
            print(f"  [AVISO] {path}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"{len(X)} imagens processadas, {X.shape[1]} atributos por imagem "
          f"(histograma RGB+HSV, sem nenhuma rede neural)\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=300, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')

    print(f"F1-macro do baseline clássico (RandomForest sobre histograma de cor): "
          f"{scores.mean():.4f} +- {scores.std():.4f}")
    print(f"F1-macro dos modelos profundos (referência): ~0.97-0.99")
    print(f"Acaso ({NUM_CLASSES} classes): ~{1/NUM_CLASSES:.4f}\n")

    gap = 0.97 - scores.mean()
    if scores.mean() > 0.85:
        print("[ALERTA] Baseline clássico chega muito perto dos modelos profundos. "
              "Isso sugere que a tarefa, nesse dataset específico, é resolvível em "
              "grande parte por sinal raso de cor/textura — o resultado alto dos "
              "modelos profundos pode refletir facilidade da tarefa mais do que "
              "capacidade de aprender morfologia fina.\n")
        print("Investigando se o sinal vem de fundo/solo (proxy de fonte de coleta) "
              "em vez da planta em si...\n")

        # Diagnóstico 1: cor prediz FONTE tão bem quanto prediz classe?
        fontes = []
        for path in kfold['test_paths']:
            stem = Path(path).stem
            if stem.startswith('usuario_'):
                fontes.append('usuario')
            elif stem.startswith('professor_'):
                fontes.append('professor')
            elif stem.startswith('externo_'):
                fontes.append('externo')
            else:
                fontes.append('desconhecida')
        fontes = np.array(fontes)[:len(X)]

        scores_fonte = cross_val_score(clf, X, fontes, cv=skf, scoring='f1_macro')
        print(f"F1-macro de cor prevendo FONTE (3 fontes, acaso ~0.33): "
              f"{scores_fonte.mean():.4f} +- {scores_fonte.std():.4f}")

        # Diagnóstico 2: baseline funciona melhor em classes puras (1 fonte só)
        # do que em classes mistura (múltiplas fontes)?
        classes_puras_nomes = {'R3_R4', 'R5', 'V12', 'V2', 'V6'}
        idx_classe_nome = {v: k for k, v in LABEL_TO_IDX.items()}
        mascara_pura = np.array([idx_classe_nome[label] in classes_puras_nomes for label in y])

        if mascara_pura.sum() > 20 and (~mascara_pura).sum() > 20:
            scores_puras = cross_val_score(clf, X[mascara_pura], y[mascara_pura], cv=5, scoring='f1_macro')
            scores_mistas = cross_val_score(clf, X[~mascara_pura], y[~mascara_pura], cv=5, scoring='f1_macro')
            print(f"\nF1-macro do baseline SÓ nas classes puras (1 fonte só): "
                  f"{scores_puras.mean():.4f} +- {scores_puras.std():.4f}")
            print(f"F1-macro do baseline SÓ nas classes mistura (múltiplas fontes): "
                  f"{scores_mistas.mean():.4f} +- {scores_mistas.std():.4f}")

            if scores_puras.mean() - scores_mistas.mean() > 0.15:
                print("\n[CONFIRMADO] Baseline funciona muito melhor nas classes puras — "
                      "forte indício de que o sinal de cor é proxy de FONTE (sessão de "
                      "coleta, solo, câmera), não fenologia real da planta.")
            else:
                print("\nBaseline funciona de forma parecida em ambos os grupos — o sinal "
                      "de cor não parece depender fortemente de quantas fontes compõem a "
                      "classe, o que enfraquece a hipótese de confundidor por fonte "
                      "(pode ser sinal de cor real da planta, ou outro confundidor não "
                      "capturado por essa quebra).")
    elif scores.mean() > 0.6:
        print(f"Baseline moderado — captura parte real do sinal (provavelmente cor de "
              f"solo/vegetação correlacionada com estádio), mas fica {gap:.2f} atrás dos "
              f"modelos profundos. Parte do desempenho profundo é sinal raso capturável, "
              f"parte é capacidade real de discriminação fina — proporção razoável para "
              f"reportar sem alarme, mas vale mencionar como contexto.")
    else:
        print(f"Baseline fica bem abaixo dos modelos profundos ({gap:.2f} de diferença) — "
              f"evidência de que o desempenho alto não vem de sinal raso de cor/textura "
              f"capturável por método clássico. Reforça que os modelos estão aprendendo "
              f"algo além de estatística superficial de imagem.")

    # Teste separado, independente do resultado acima: cor SÓ da planta
    # (ignorando fundo/solo) prevê fonte tão bem quanto prevê classe? Esse
    # teste não sofre do problema do anterior (classe e fonte confundidas
    # por construção do dataset) porque isola o que é da planta do que é
    # da cena — se ainda assim previr fonte tão bem quanto classe, é
    # confundidor real na própria cor da folha, não só no fundo.
    print(f"\n{'=' * 60}\nCor só de vegetação (ignora fundo/solo)\n{'=' * 60}")
    print("Extraindo histograma só dos pixels de planta (máscara de verde crua)...")

    X_veg = []
    for path in kfold['test_paths'][:len(X)]:
        try:
            X_veg.append(extrair_features_vegetacao(path))
        except Exception:
            X_veg.append(np.zeros(3 * BINS_HISTOGRAMA + 7))
    X_veg = np.array(X_veg)

    fontes = []
    for path in kfold['test_paths'][:len(X)]:
        stem = Path(path).stem
        if stem.startswith('usuario_'):
            fontes.append('usuario')
        elif stem.startswith('professor_'):
            fontes.append('professor')
        elif stem.startswith('externo_'):
            fontes.append('externo')
        else:
            fontes.append('desconhecida')
    fontes = np.array(fontes)

    scores_classe_veg = cross_val_score(clf, X_veg, y, cv=skf, scoring='f1_macro')
    scores_fonte_veg = cross_val_score(clf, X_veg, fontes, cv=skf, scoring='f1_macro')

    print(f"\nF1-macro prevendo CLASSE com cor só de vegetação: "
          f"{scores_classe_veg.mean():.4f} +- {scores_classe_veg.std():.4f}")
    print(f"F1-macro prevendo FONTE com cor só de vegetação: "
          f"{scores_fonte_veg.mean():.4f} +- {scores_fonte_veg.std():.4f}")

    diferenca = scores_classe_veg.mean() - scores_fonte_veg.mean()
    if scores_fonte_veg.mean() > 0.7:
        print(f"\n[CONFIRMADO] Cor só da planta ainda prevê fonte bem "
              f"({scores_fonte_veg.mean():.4f}) — confundidor real, não só de fundo/solo. "
              f"Câmera/processamento deixa marca de cor até na folha. Precisa de correção "
              f"(variação de matiz na augmentation) antes de tratar os resultados como finais.")
    elif diferenca > 0.2:
        print(f"\nCor de vegetação prevê CLASSE bem melhor que FONTE (diferença de "
              f"{diferenca:.4f}) — evidência a favor de que o sinal de cor é real "
              f"biologia da planta (muda com o estádio), não confundidor de fonte. "
              f"O alerta do teste de cena inteira era, em boa parte, sobre fundo/solo, "
              f"não sobre a planta em si.")
    else:
        print(f"\nResultado misto — não dá pra afirmar com confiança que o sinal é "
              f"biológico nem que é confundidor. Recomendo tratar como incerteza "
              f"registrada, não como conclusão fechada.")


if __name__ == "__main__":
    main()
