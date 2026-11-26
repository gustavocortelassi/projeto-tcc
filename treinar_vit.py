"""
Treino K-Fold do modelo vit.

Casca fina sobre treino_core.py — checkpoint por época, retomada automática
e resultado incremental por fold estão lá, não aqui. Ver treino_core.py
antes de alterar qualquer coisa no comportamento de treino.

Uso:
    python treinar_vit.py
"""

from treino_core import treinar_modelo

if __name__ == "__main__":
    treinar_modelo('vit')
