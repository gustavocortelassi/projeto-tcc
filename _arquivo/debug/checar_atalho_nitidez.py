"""
Verifica se a nitidez da imagem prediz a fonte de origem (usuário,
professor ou banco externo) — e por extensão a classe, nas classes que
vierem de uma única fonte só.

Usa variância do Laplaciano — métrica padrão e simples de nitidez/blur:
quanto menor a variância, mais borrada a imagem. Mesmo princípio do
checar_atalho_bordas.py, aplicado a um sinal diferente.

Uso:
    python checar_atalho_nitidez.py
"""

import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
from PIL import Image, ImageFile
from scipy.signal import convolve2d
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

PASTA_UNIFICADA = Path("dataset_unificado")
LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def variancia_laplaciano(path):
    img = np.array(Image.open(path).convert('L'), dtype=np.float64)
    lap = convolve2d(img, LAPLACIAN_KERNEL, mode='valid')
    return lap.var()


def fonte_do_arquivo(path):
    if path.stem.startswith('usuario_'):
        return 'usuario'
    if path.stem.startswith('professor_'):
        return 'professor'
    if path.stem.startswith('externo_'):
        return 'externo'
    return 'desconhecida'


def main():
    if not PASTA_UNIFICADA.exists():
        print(f"[ERRO] Pasta não encontrada: {PASTA_UNIFICADA}")
        return

    dados = []
    for cls_dir in sorted(p for p in PASTA_UNIFICADA.iterdir() if p.is_dir()):
        for arquivo in cls_dir.iterdir():
            if arquivo.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            try:
                nitidez = variancia_laplaciano(arquivo)
            except Exception:
                continue
            dados.append({
                'classe': cls_dir.name,
                'fonte': fonte_do_arquivo(arquivo),
                'nitidez': nitidez,
            })

    if not dados:
        print("Nenhuma imagem encontrada.")
        return

    print(f"{'Classe':<10} {'Fonte':<12} {'n':>6} {'Nitidez média':>15} {'Desvio':>10}")
    por_classe_fonte = defaultdict(list)
    for d in dados:
        por_classe_fonte[(d['classe'], d['fonte'])].append(d['nitidez'])

    for (classe, fonte), valores in sorted(por_classe_fonte.items()):
        media = np.mean(valores)
        desvio = np.std(valores)
        print(f"{classe:<10} {fonte:<12} {len(valores):>6} {media:>15.1f} {desvio:>10.1f}")

    # Teste 1: nitidez sozinha prediz a FONTE?
    X = np.array([[d['nitidez']] for d in dados])
    y_fonte = np.array([d['fonte'] for d in dados])
    fontes_presentes = sorted(set(y_fonte))
    acaso_fonte = 1 / len(fontes_presentes)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    scores_fonte = cross_val_score(clf, X, y_fonte, cv=5, scoring='f1_macro')
    print(f"\nF1-macro prevendo FONTE só com nitidez ({len(fontes_presentes)} fontes: "
          f"{fontes_presentes}): {scores_fonte.mean():.4f} +- {scores_fonte.std():.4f}")
    print(f"(Acaso, {len(fontes_presentes)} fontes: ~{acaso_fonte:.4f})")

    # Teste 2: cada par de fonte, binário — mais sensível que o agregado de
    # N vias (mesma lição da borda: agregado pode esconder separação clara
    # entre um par específico)
    print("\nComparações binárias por par de fonte:")
    for f1, f2 in combinations(fontes_presentes, 2):
        dados_par = [d for d in dados if d['fonte'] in (f1, f2)]
        X_par = np.array([[d['nitidez']] for d in dados_par])
        y_par = np.array([d['fonte'] for d in dados_par])
        scores_par = cross_val_score(clf, X_par, y_par, cv=5, scoring='f1_macro')
        alerta = "  [ALERTA]" if scores_par.mean() > 0.8 else ""
        print(f"  {f1:<12} vs {f2:<12} F1={scores_par.mean():.4f} +- {scores_par.std():.4f}{alerta}")

    # Teste 3: classes 100% de uma fonte só — descobertas automaticamente a
    # partir do dado real, não presumidas (a lista mudou com a Fonte 3)
    classes_por_fonte = defaultdict(set)
    for d in dados:
        classes_por_fonte[d['classe']].add(d['fonte'])
    classes_puras = {c: list(fs)[0] for c, fs in classes_por_fonte.items() if len(fs) == 1}

    if not classes_puras:
        print("\nNenhuma classe vem de uma única fonte — toda classe é mistura. "
              "Risco de atalho por classe pura não se aplica nesse dataset.")
        return

    print(f"\nClasses 100% de uma única fonte: {classes_puras}")
    dados_puros = [d for d in dados if d['classe'] in classes_puras]
    X_puro = np.array([[d['nitidez']] for d in dados_puros])
    y_puro = np.array([d['classe'] for d in dados_puros])
    scores_puro = cross_val_score(clf, X_puro, y_puro, cv=5, scoring='f1_macro')
    acaso_puro = 1 / len(set(y_puro))
    print(f"F1-macro prevendo CLASSE só com nitidez, nas classes puras: "
          f"{scores_puro.mean():.4f} +- {scores_puro.std():.4f} (acaso ~{acaso_puro:.4f})")

    if scores_puro.mean() > 2 * acaso_puro:
        print("\n[ALERTA] Nitidez prediz classe pura bem acima do acaso. Atalho residual "
              "possível nas classes que vêm de fonte única — considerar como limitação "
              "explícita ou investigar correção antes de reportar como final.")
    else:
        print("\nF1 agregado nas classes puras não confirma atalho — mas isso pode esconder "
              "separação clara de uma classe específica contra o resto, como já aconteceu "
              "antes. Testando cada classe pura individualmente contra as demais.")

    print("\nCada classe pura vs. o resto das classes puras (um-contra-resto):")
    for classe_alvo in classes_puras:
        y_binario = np.array([classe_alvo if c == classe_alvo else 'outras' for c in y_puro])
        if len(set(y_binario)) < 2:
            continue
        scores_bin = cross_val_score(clf, X_puro, y_binario, cv=5, scoring='f1_macro')
        alerta = "  [ALERTA]" if scores_bin.mean() > 0.8 else ""
        print(f"  {classe_alvo:<10} ({classes_puras[classe_alvo]}) vs. resto: "
              f"F1={scores_bin.mean():.4f} +- {scores_bin.std():.4f}{alerta}")


if __name__ == "__main__":
    main()
