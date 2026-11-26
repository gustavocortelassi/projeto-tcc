"""
Gera relatório visual completo a partir dos resultados já salvos em
outputs/results/ e outputs/logs/ — não treina nada, só lê e plota.

Estrutura de saída em outputs/plots/relatorio/:
  comparativo/
    f1_comparativo.png       - ranking de F1-macro dos 5 modelos
    f1_por_classe_heatmap.png - mapa de calor F1 por classe x modelo
  <modelo>/                  - uma pasta por arquitetura, com:
    f1_por_classe.png        - F1 por classe, só esse modelo
    matriz_confusao.png      - matriz de confusão, só esse modelo
    variancia_folds.png      - F1 de teste por fold, só esse modelo
    curva_treino.png         - val_f1 por época, uma linha por fold

Uso:
    python 09_gerar_relatorio_visual.py
"""

import sys
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import *

OUT_DIR_REL = PLOT_DIR / 'relatorio'
LOGS_DIR = PROJECT_ROOT / 'logs'

plt.rcParams['figure.dpi'] = 120


def carregar_comparativo():
    path = RESULT_DIR / 'comparativo_final.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


def carregar_avaliacoes():
    avaliacoes = {}
    for modelo in MODELS:
        path = RESULT_DIR / f'avaliacao_{modelo}.json'
        if not path.exists():
            continue
        with open(path, encoding='utf-8') as f:
            avaliacoes[modelo] = json.load(f)
    return avaliacoes


def carregar_resultados_fold():
    resultados = {}
    for modelo in MODELS:
        path = RESULT_DIR / f'resultado_{modelo}.json'
        if not path.exists():
            continue
        with open(path, encoding='utf-8') as f:
            resultados[modelo] = json.load(f)
    return resultados


def parsear_log(modelo):
    """Retorna {fold: [(epoca, val_f1), ...]} a partir do log de treino."""
    path = LOGS_DIR / f'{modelo}.log'
    if not path.exists():
        return {}

    with open(path, encoding='utf-8', errors='replace') as f:
        texto = f.read()

    folds = {}
    fold_atual = None
    for linha in texto.splitlines():
        m_fold = re.search(r'Fold\s+(\d+)/\d+', linha)
        if m_fold:
            fold_atual = int(m_fold.group(1))
            folds.setdefault(fold_atual, [])
            continue
        m_epoca = re.search(r'Epoch\s+(\d+).*?val_f1:\s*([\d.]+)', linha)
        if m_epoca and fold_atual is not None:
            folds[fold_atual].append((int(m_epoca.group(1)), float(m_epoca.group(2))))

    return folds


# ─── Gráficos comparativos (todos os modelos juntos) ────────────────────────

def comparativo_f1(df, destino_dir):
    if df is None:
        print("  comparativo_final.csv não encontrado, pulando f1_comparativo")
        return

    df = df.sort_values('f1_macro_ensemble_teste', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['f1_macro_ensemble_teste'], color='#4472C4')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['modelo'])
    ax.set_xlabel('F1-macro (ensemble, conjunto de teste)')
    ax.set_xlim(0.9, 1.0)
    ax.set_title('Comparativo de F1-macro por arquitetura')
    for i, v in enumerate(df['f1_macro_ensemble_teste']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(destino_dir / 'f1_comparativo.png')
    plt.close(fig)
    print("  comparativo/f1_comparativo.png salvo")


def comparativo_heatmap(avaliacoes, destino_dir):
    if not avaliacoes:
        print("  nenhuma avaliacao_<modelo>.json encontrada, pulando heatmap")
        return

    modelos = list(avaliacoes.keys())
    classes = avaliacoes[modelos[0]]['classes']
    matriz = np.zeros((len(modelos), len(classes)))

    for i, modelo in enumerate(modelos):
        rep = avaliacoes[modelo]['classification_report']
        for j, cls in enumerate(classes):
            matriz[i, j] = rep[cls]['f1-score']

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matriz, cmap='RdYlGn', vmin=0.8, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(modelos)))
    ax.set_yticklabels(modelos)
    ax.set_title('F1-score por classe e por modelo')
    for i in range(len(modelos)):
        for j in range(len(classes)):
            ax.text(j, i, f'{matriz[i, j]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='F1-score')
    plt.tight_layout()
    plt.savefig(destino_dir / 'f1_por_classe_heatmap.png')
    plt.close(fig)
    print("  comparativo/f1_por_classe_heatmap.png salvo")


# ─── Gráficos por arquitetura ────────────────────────────────────────────────

def modelo_f1_por_classe(modelo, avaliacao, destino_dir):
    classes = avaliacao['classes']
    rep = avaliacao['classification_report']
    valores = [rep[c]['f1-score'] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4))
    cores = ['#C0504D' if v < 0.95 else '#4472C4' for v in valores]
    ax.bar(classes, valores, color=cores)
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel('F1-score')
    ax.set_title(f'F1 por classe — {modelo}')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(valores):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(destino_dir / 'f1_por_classe.png')
    plt.close(fig)


def modelo_matriz_confusao(modelo, avaliacao, destino_dir):
    classes = avaliacao['classes']
    cm = np.array(avaliacao['confusion_matrix'])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title(f'Matriz de confusão — {modelo}')

    for i in range(len(classes)):
        for j in range(len(classes)):
            valor = cm_norm[i, j]
            if valor < 0.005:  # esconde zero/quase-zero, deixa a matriz legível
                continue
            cor_texto = 'white' if valor > 0.5 else 'black'
            ax.text(j, i, f'{valor:.2f}', ha='center', va='center',
                    fontsize=6, color=cor_texto)

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(destino_dir / 'matriz_confusao.png')
    plt.close(fig)


def modelo_variancia_folds(modelo, resultado, destino_dir):
    f1s = resultado['test_f1_por_fold']
    fig, ax = plt.subplots(figsize=(5, 5))
    x = [0] * len(f1s)
    ax.scatter(x, f1s, s=100, alpha=0.8, zorder=3, color='#4472C4')
    for i, v in enumerate(f1s):
        ax.text(0.05, v, f'fold {i+1}: {v:.4f}', va='center', fontsize=9)
    media = np.mean(f1s)
    desvio = np.std(f1s)
    ax.hlines(media, -0.3, 0.3, color='black', linewidth=1.5, zorder=4,
              label=f'média={media:.4f}, desvio={desvio:.4f}')
    ax.set_xlim(-0.5, 1.3)
    ax.set_xticks([])
    ax.set_ylabel('F1-macro de teste')
    ax.set_title(f'Variância entre folds — {modelo}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(destino_dir / 'variancia_folds.png')
    plt.close(fig)


def modelo_curva_treino(modelo, destino_dir):
    log_path = LOGS_DIR / f'{modelo}.log'
    folds = parsear_log(modelo)
    if not folds:
        print(f"    log não encontrado ou sem linha de época reconhecida: {log_path}")
        print(f"    (existe no disco: {log_path.exists()})")
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for fold_idx, pontos in sorted(folds.items()):
        if not pontos:
            continue
        epocas, f1s = zip(*pontos)
        ax.plot(epocas, f1s, marker='o', markersize=3, label=f'Fold {fold_idx}')

    ax.set_xlabel('Época')
    ax.set_ylabel('val_f1')
    ax.set_title(f'Curva de treino — {modelo}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(destino_dir / 'curva_treino.png')
    plt.close(fig)
    return True


def main():
    OUT_DIR_REL.mkdir(parents=True, exist_ok=True)
    print(f"Gerando relatório em: {OUT_DIR_REL}\n")

    df = carregar_comparativo()
    avaliacoes = carregar_avaliacoes()
    resultados = carregar_resultados_fold()

    destino_comparativo = OUT_DIR_REL / 'comparativo'
    destino_comparativo.mkdir(parents=True, exist_ok=True)
    comparativo_f1(df, destino_comparativo)
    comparativo_heatmap(avaliacoes, destino_comparativo)

    for modelo in MODELS:
        destino_modelo = OUT_DIR_REL / modelo
        destino_modelo.mkdir(parents=True, exist_ok=True)
        print(f"\n{modelo}/")

        if modelo in avaliacoes:
            modelo_f1_por_classe(modelo, avaliacoes[modelo], destino_modelo)
            modelo_matriz_confusao(modelo, avaliacoes[modelo], destino_modelo)
            print(f"  {modelo}/f1_por_classe.png e matriz_confusao.png salvos")
        else:
            print(f"  avaliacao_{modelo}.json não encontrada, pulando")

        if modelo in resultados:
            modelo_variancia_folds(modelo, resultados[modelo], destino_modelo)
            print(f"  {modelo}/variancia_folds.png salvo")
        else:
            print(f"  resultado_{modelo}.json não encontrada, pulando variancia_folds")

        if modelo_curva_treino(modelo, destino_modelo):
            print(f"  {modelo}/curva_treino.png salvo")

    print(f"\nRelatório completo em: {OUT_DIR_REL}")


if __name__ == "__main__":
    main()
