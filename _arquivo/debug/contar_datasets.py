"""
Varre o projeto procurando pastas de dataset (pastas cujas subpastas têm
nome de classe fenológica) e imprime a contagem de imagens por classe em
cada uma. Somente leitura: não altera nenhum arquivo.

Uso:
    python contar_datasets.py
"""

from pathlib import Path

RAIZ = Path(".")
CLASSES_CONHECIDAS = {
    'R1', 'R2', 'R3', 'R4', 'R3_R4', 'R5', 'R6',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V4_V5', 'V6', 'V7', 'V7_V8', 'V8',
    'V12', 'VT', 'VE',
}
EXT_IMAGEM = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
IGNORAR = {'.git', '__pycache__', 'models', '.ipynb_checkpoints'}
MIN_CLASSES = 3  # pasta precisa ter ao menos 3 subpastas de classe


def contar_imagens(pasta):
    return sum(1 for f in pasta.iterdir()
               if f.is_file() and f.suffix.lower() in EXT_IMAGEM)


def main():
    encontradas = 0
    for pasta in sorted(RAIZ.rglob('*')):
        if not pasta.is_dir() or any(p in IGNORAR for p in pasta.parts):
            continue
        subs = [d for d in pasta.iterdir()
                if d.is_dir() and d.name.upper() in CLASSES_CONHECIDAS]
        if len(subs) < MIN_CLASSES:
            continue

        encontradas += 1
        total = 0
        print(f"\n=== {pasta} ===")
        for d in sorted(subs, key=lambda x: x.name):
            n = contar_imagens(d)
            total += n
            print(f"  {d.name:8} {n:>6}")
        print(f"  {'TOTAL':8} {total:>6}")

    if encontradas == 0:
        print("Nenhuma pasta de dataset encontrada a partir de", RAIZ.resolve())


if __name__ == "__main__":
    main()
