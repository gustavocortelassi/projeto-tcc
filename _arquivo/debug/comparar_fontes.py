"""
Lê duas pastas de dataset (cada uma com subpastas por classe), lista as
classes de cada uma, mostra onde se sobrepõem, e traz contagem + faixa de
resolução por classe em cada fonte.

Uso:
    python comparar_fontes.py "Dataset Nova Tentativa" "Dataset Nova Tentativa Professor"
"""

import sys
from pathlib import Path
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp',
              '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP'}


def listar_classes(pasta):
    return {p.name: p for p in pasta.iterdir() if p.is_dir()}


def contar_e_medir(cls_dir):
    n = 0
    larguras, alturas = [], []
    for f in cls_dir.iterdir():
        if f.suffix not in EXTENSIONS:
            continue
        try:
            with Image.open(f) as img:
                w, h = img.size
            larguras.append(w)
            alturas.append(h)
            n += 1
        except Exception:
            continue
    return n, larguras, alturas


def resumo_classe(cls_dir):
    n, larguras, alturas = contar_e_medir(cls_dir)
    if n == 0:
        return f"{n:>5} imagens"
    lmin, lmax = min(larguras), max(larguras)
    amin, amax = min(alturas), max(alturas)
    return f"{n:>5} imagens | largura {lmin}-{lmax}px | altura {amin}-{amax}px"


def main():
    if len(sys.argv) != 3:
        print("Uso: python comparar_fontes.py <pasta1> <pasta2>")
        return

    pasta1 = Path(sys.argv[1])
    pasta2 = Path(sys.argv[2])

    if not pasta1.exists() or not pasta2.exists():
        print(f"[ERRO] Verifique os caminhos: {pasta1} | {pasta2}")
        return

    classes1 = listar_classes(pasta1)
    classes2 = listar_classes(pasta2)

    nomes1 = set(classes1.keys())
    nomes2 = set(classes2.keys())
    comuns = sorted(nomes1 & nomes2)
    so1 = sorted(nomes1 - nomes2)
    so2 = sorted(nomes2 - nomes1)

    print(f"Pasta 1: {pasta1}  ({len(nomes1)} classes)")
    print(f"Pasta 2: {pasta2}  ({len(nomes2)} classes)\n")

    print(f"Classes em comum ({len(comuns)}):")
    for cls in comuns:
        print(f"\n  {cls}")
        print(f"    Pasta 1: {resumo_classe(classes1[cls])}")
        print(f"    Pasta 2: {resumo_classe(classes2[cls])}")

    if so1:
        print(f"\nSó na Pasta 1 ({len(so1)}): {so1}")
    if so2:
        print(f"\nSó na Pasta 2 ({len(so2)}): {so2}")


if __name__ == "__main__":
    main()
