"""
Confirma que a variação de matiz foi aplicada de verdade nas imagens de
treino (dataset_split/augmented/) — não no teste, que nunca passa por
augmentation por design.

Mede a dispersão (desvio padrão circular) do matiz médio por imagem,
agrupado por fonte. Se o jitter funcionou, o desvio dentro de cada fonte
deve estar bem maior que antes (a marca de câmera, antes um valor quase
fixo por fonte, agora devia estar espalhada pelo jitter de ±18°).

Uso:
    python 11_checar_jitter_aplicado.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).parent))
from config import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def matiz_medio(path):
    hsv = Image.open(path).convert('HSV').resize((64, 64))
    h = np.array(hsv.split()[0], dtype=np.float64)
    angulos = h / 256 * 2 * np.pi
    media_sin = np.sin(angulos).mean()
    media_cos = np.cos(angulos).mean()
    media_circular = np.arctan2(media_sin, media_cos) % (2 * np.pi)
    return media_circular / (2 * np.pi) * 360


def fonte_do_arquivo(nome):
    if 'usuario_' in nome:
        return 'usuario'
    if 'professor_' in nome:
        return 'professor'
    if 'externo_' in nome:
        return 'externo'
    return 'desconhecida'


def main():
    print(f"Lendo imagens de: {AUGMENTED_DIR}\n")

    matizes_por_fonte = defaultdict(list)
    total = 0

    for cls_dir in AUGMENTED_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
        for arquivo in cls_dir.iterdir():
            fonte = fonte_do_arquivo(arquivo.name)
            try:
                m = matiz_medio(arquivo)
                matizes_por_fonte[fonte].append(m)
                total += 1
            except Exception:
                continue

    print(f"{total} imagens processadas\n")
    print(f"{'Fonte':<15} {'n':>6} {'Matiz médio':>14} {'Desvio':>10}")
    for fonte, valores in sorted(matizes_por_fonte.items()):
        media = np.mean(valores)
        desvio = np.std(valores)
        print(f"{fonte:<15} {len(valores):>6} {media:>13.1f} graus {desvio:>9.1f} graus")

    print(f"\nSe o jitter (+-18 graus) foi aplicado corretamente, o desvio dentro de "
          f"cada fonte deve estar em torno de 10-15 graus (desvio de uma distribuicao "
          f"uniforme de +-18 graus e ~10.4) -- bem maior que a variacao natural de "
          f"fotos sem jitter nenhum, que tende a ser mais estreita e concentrada na "
          f"cor real da vegetacao (verde, ~90-150 graus).")


if __name__ == "__main__":
    main()
