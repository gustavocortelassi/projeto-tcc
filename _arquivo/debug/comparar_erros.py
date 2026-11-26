"""
Lê as matrizes de confusão de outputs/results/avaliacao_<modelo>.json e lista
os erros (fora da diagonal) de cada modelo, mais a comparação entre modelos.
Somente leitura.

Uso:
    python comparar_erros.py
"""

import json
from pathlib import Path

RESULT_DIR = Path("outputs/results")
MODELOS = ['mobilevit', 'efficientnetv2', 'resnet50', 'vit', 'swin']


def erros_de(modelo):
    with open(RESULT_DIR / f"avaliacao_{modelo}.json", encoding="utf-8") as f:
        d = json.load(f)
    cls, cm = d['classes'], d['confusion_matrix']
    return {(cls[i], cls[j]): cm[i][j]
            for i in range(len(cls)) for j in range(len(cls)) if i != j and cm[i][j] > 0}


def main():
    todos = {}
    for m in MODELOS:
        try:
            todos[m] = erros_de(m)
        except FileNotFoundError:
            print(f"{m}: avaliacao_{m}.json não encontrado")

    for m, err in todos.items():
        print(f"\n{m} ({sum(err.values())} erros)")
        for (real, pred), n in sorted(err.items(), key=lambda x: -x[1]):
            print(f"  {real:>6} -> {pred:<6} {n}")

    pares = sorted({p for e in todos.values() for p in e})
    print(f"\n{'Real -> Previsto':18}" + "".join(f"{m[:9]:>11}" for m in todos))
    for p in pares:
        print(f"{p[0] + ' -> ' + p[1]:18}" + "".join(f"{todos[m].get(p, 0):>11}" for m in todos))


if __name__ == "__main__":
    main()
