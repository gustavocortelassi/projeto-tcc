"""
Investigação somente leitura do pipeline. Não altera nenhum arquivo.

1. Lista os scripts .py da raiz (ordem do pipeline)
2. Mostra onde V7_V8 / V8 / V4_V5 / EXCLUDED_CLASSES / CLASS_MAP aparecem no código
3. Lista arquivos de resultado/log (.json .csv .txt .log) com tamanho e data

Uso:
    python investigar_pipeline.py > investigacao.txt
"""

import re
from datetime import datetime
from pathlib import Path

RAIZ = Path(".")
IGNORAR = {'.git', '__pycache__', '.ipynb_checkpoints', 'models',
           'dataset_split', 'dataset_unificado'}
TERMOS = re.compile(r"V7_V8|V4_V5|\bV8\b|EXCLUDED_CLASSES|CLASS_MAP|CLASS_ORIGEM")
EXT_LOG = {'.json', '.csv', '.txt', '.log'}
MAX_ARQUIVOS_LOG = 80


def ignorado(p):
    return any(parte in IGNORAR for parte in p.parts)


def secao(titulo):
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")


def main():
    # 1. Scripts da raiz
    secao("1. SCRIPTS .py NA RAIZ")
    for f in sorted(RAIZ.glob("*.py")):
        data = datetime.fromtimestamp(f.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
        print(f"  {f.name:45} {data}")

    # 2. Ocorrências no código (raiz + subpastas)
    secao("2. OCORRÊNCIAS DE V7_V8 / V8 / V4_V5 / EXCLUDED_CLASSES / CLASS_MAP")
    for f in sorted(RAIZ.rglob("*.py")):
        if ignorado(f) or f.name == Path(__file__).name:
            continue
        try:
            linhas = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        achados = [(i, l.strip()) for i, l in enumerate(linhas, 1) if TERMOS.search(l)]
        if achados:
            print(f"\n  --- {f} ---")
            for i, l in achados:
                print(f"    L{i:<5} {l[:110]}")

    # 3. Arquivos de resultado / log
    secao("3. ARQUIVOS DE RESULTADO E LOG")
    arquivos = [f for f in RAIZ.rglob("*")
                if f.is_file() and f.suffix.lower() in EXT_LOG and not ignorado(f)]
    arquivos.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    for f in arquivos[:MAX_ARQUIVOS_LOG]:
        st = f.stat()
        data = datetime.fromtimestamp(st.st_mtime).strftime("%d/%m/%Y %H:%M")
        print(f"  {data}  {st.st_size / 1024:>8.1f} KB  {f}")
    if len(arquivos) > MAX_ARQUIVOS_LOG:
        print(f"  ... e mais {len(arquivos) - MAX_ARQUIVOS_LOG} arquivos (mais antigos)")


if __name__ == "__main__":
    main()
