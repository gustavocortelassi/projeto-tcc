"""
Teste de significância entre os 5 modelos, usando F1 de teste por fold —
mesmos 5 folds para todos os modelos, então é comparação pareada.

N=5 é pequeno — poder estatístico baixo por natureza do desenho (K-Fold com
5 folds, não uma escolha arbitrária de N). Testes não-paramétricos
(Friedman + Wilcoxon pareado) são mais apropriados aqui que ANOVA/t-test,
mas mesmo assim o resultado é evidência a mais, não prova isolada — reportar
como tal.

Uso:
    python 06_teste_significancia.py
"""

import sys
import json
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def carregar_f1_por_fold():
    dados = {}
    for modelo in MODELS:
        path = RESULT_DIR / f"resultado_{modelo}.json"
        if not path.exists():
            continue
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        dados[modelo] = r['test_f1_por_fold']
    return dados


def main():
    dados = carregar_f1_por_fold()
    modelos = list(dados.keys())
    if len(modelos) < 3:
        print("Precisa de pelo menos 3 modelos com resultado_<modelo>.json para o teste de Friedman.")
        return

    print("F1 de teste por fold:\n")
    for m in modelos:
        print(f"  {m:<16} {[round(v, 4) for v in dados[m]]}")

    matriz = np.array([dados[m] for m in modelos])

    stat, p = stats.friedmanchisquare(*matriz)
    print(f"\nFriedman (H0: sem diferença entre as {len(modelos)} arquiteturas): "
          f"chi2={stat:.3f}, p={p:.4f}")
    if p < 0.05:
        print("  -> rejeita H0 a 5%: há diferença entre pelo menos um par de arquiteturas.")
    else:
        print("  -> não rejeita H0 a 5%: com N=5 folds, não há evidência estatística "
              "suficiente de diferença — mesmo que o ranking pareça claro, a amostra "
              "é pequena demais para confirmar isso sozinha.")

    n_comparacoes = len(list(combinations(modelos, 2)))
    alfa_bonferroni = 0.05 / n_comparacoes
    print(f"\nComparações pareadas (Wilcoxon signed-rank, mesmos folds). "
          f"{n_comparacoes} comparações -> alfa ajustado (Bonferroni) = {alfa_bonferroni:.4f}:\n")

    for m1, m2 in combinations(modelos, 2):
        try:
            stat_w, p_w = stats.wilcoxon(dados[m1], dados[m2])
            marca = "*** (p<0.05)" if p_w < 0.05 else ""
            marca_bonf = " **(sobrevive a Bonferroni)**" if p_w < alfa_bonferroni else ""
            print(f"  {m1:<16} vs {m2:<16} p={p_w:.4f} {marca}{marca_bonf}")
        except ValueError:
            print(f"  {m1:<16} vs {m2:<16} não computável — diferenças idênticas em todos os folds")

    print("\nAviso: com 5 folds, o Wilcoxon pareado tem piso de p em torno de 0,06 mesmo "
          "no caso de diferença perfeita em todos os folds — não espere p<0,05 sem uma "
          "diferença bem consistente entre os 5 folds. Ausência de significância aqui "
          "não é prova de equivalência entre arquiteturas, é ausência de evidência "
          "suficiente com essa amostra. Reportar como limitação explícita no TCC.")


if __name__ == "__main__":
    main()
