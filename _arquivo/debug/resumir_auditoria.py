"""
Resume os JSONs de auditoria (estrutura + contagens), sem imprimir o
conteúdo inteiro. Somente leitura.

Uso:
    python resumir_auditoria.py > resumo_auditoria.txt
"""

import json
from collections import Counter
from pathlib import Path

ARQUIVOS = [
    Path("objetos_filtrados.json"),
    Path("grupos_quase_duplicata.json"),
    Path("outputs/results/excedente.json"),
]
MAX_VALORES_DISTINTOS = 30   # campo com até N valores distintos vira contagem
AMOSTRA_STR = 100


def amostra(v):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= AMOSTRA_STR else s[:AMOSTRA_STR] + "..."


def resumir_lista_de_dicts(lista, indent):
    pad = " " * indent
    chaves = Counter(k for item in lista if isinstance(item, dict) for k in item)
    print(f"{pad}{len(lista)} itens | chaves: {dict(chaves)}")
    print(f"{pad}exemplo: {amostra(lista[0])}")
    for chave in chaves:
        valores = [item.get(chave) for item in lista if isinstance(item, dict)]
        hashaveis = [v for v in valores if isinstance(v, (str, int, bool, type(None)))]
        if not hashaveis:
            continue
        cont = Counter(hashaveis)
        if len(cont) <= MAX_VALORES_DISTINTOS:
            print(f"{pad}  contagem de '{chave}':")
            for val, n in cont.most_common():
                print(f"{pad}    {str(val):30} {n:>7}")


def resumir(obj, indent=2, profundidade=0):
    pad = " " * indent
    if profundidade > 3:
        print(f"{pad}(profundidade máxima)")
        return
    if isinstance(obj, dict):
        print(f"{pad}dict com {len(obj)} chaves")
        for i, (k, v) in enumerate(obj.items()):
            if i >= 25:
                print(f"{pad}... e mais {len(obj) - 25} chaves")
                break
            tipo = type(v).__name__
            tam = f" (len {len(v)})" if isinstance(v, (list, dict)) else ""
            print(f"{pad}- {k}: {tipo}{tam}" + ("" if tam else f" = {amostra(v)}"))
            if isinstance(v, (list, dict)) and len(obj) <= 25:
                resumir(v, indent + 4, profundidade + 1)
    elif isinstance(obj, list):
        if not obj:
            print(f"{pad}lista vazia")
        elif all(isinstance(i, dict) for i in obj[:50]):
            resumir_lista_de_dicts(obj, indent)
        else:
            print(f"{pad}lista com {len(obj)} itens | exemplo: {amostra(obj[0])}")


def main():
    for arq in ARQUIVOS:
        print(f"\n{'=' * 70}\n{arq}\n{'=' * 70}")
        if not arq.exists():
            print("  (não encontrado)")
            continue
        try:
            with open(arq, encoding="utf-8") as f:
                dados = json.load(f)
        except Exception as e:
            print(f"  [ERRO ao ler] {e}")
            continue
        resumir(dados)


if __name__ == "__main__":
    main()
