"""
Treino K-Fold de um modelo do benchmark, via linha de comando.

Uso:
    python 03_treinar.py --modelo resnet50
    python 03_treinar.py --modelo swin

Checkpoint por época, retomada automática e resultado incremental por fold
estão em treino_core.py — este arquivo só faz o parse do argumento.
"""

import argparse
from treino_core import treinar_modelo, MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelo', required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()
    treinar_modelo(args.modelo)


if __name__ == "__main__":
    main()
