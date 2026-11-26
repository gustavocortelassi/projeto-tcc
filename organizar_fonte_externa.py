"""
Organiza o banco externo (train_set, validation_set, test_set) numa
estrutura de pasta por classe, para o montar_dataset_final.py conseguir
ler como uma fonte só.

Não precisa saber se cada .zip já vem organizado em subpasta por classe ou
é uma pasta só — varre tudo recursivamente e infere a classe pelo sufixo
do nome do arquivo (ex.: "..._ZEAMX_V1.jpg" -> classe V1).

Usa os TRÊS splits juntos, de propósito — train_set e validation_set desse
banco têm sobreposição de conteúdo confirmada (mesma imagem nos dois), e a
deduplicação do montar_dataset_final.py já foi desenhada para lidar com
isso. Escolher só um dos três não evita duplicata nenhuma, só descarta
dado.

Uso:
    1. Extraia os três .zip em pastas separadas (ou uma pasta com as três
       dentro, tanto faz — o script varre recursivamente).
    2. Ajuste PASTAS_ORIGEM abaixo com os caminhos das pastas extraídas.
    3. python organizar_fonte_externa.py
"""

import sys
import shutil
from pathlib import Path

# ─── Ajuste antes de rodar ───────────────────────────────────────────────────
PASTAS_ORIGEM = [
    Path("train_set"),
    Path("validation_set"),
    Path("test_set"),
]
DESTINO = Path("Dataset V1 a V4 Externo")  # mesmo nome esperado pela Fonte 3

CLASSES_CONHECIDAS = ['R1', 'R2', 'R3_R4', 'R5', 'R6',
                       'V1', 'V2', 'V3', 'V4', 'V6', 'V7', 'V8', 'V12', 'VT']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp',
              '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP'}


def inferir_classe(nome_arquivo):
    stem = Path(nome_arquivo).stem
    partes = stem.split('_')

    # tenta os últimos 1-2 tokens contra a lista de classes conhecidas,
    # priorizando o mais específico (ex.: "R3_R4" antes de "R4" sozinho)
    for n_tokens in (2, 1):
        candidato = '_'.join(partes[-n_tokens:])
        if candidato in CLASSES_CONHECIDAS:
            return candidato
    return None


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)

    total_copiados = 0
    total_nao_classificados = 0
    nao_classificados_exemplos = []
    contagem_por_classe = {}

    for pasta_origem in PASTAS_ORIGEM:
        if not pasta_origem.exists():
            print(f"[AVISO] Pasta não encontrada, pulando: {pasta_origem}")
            continue

        print(f"Varrendo: {pasta_origem}")
        for arquivo in pasta_origem.rglob('*'):
            if not arquivo.is_file() or arquivo.suffix not in EXTENSIONS:
                continue

            classe = inferir_classe(arquivo.name)
            if classe is None:
                total_nao_classificados += 1
                if len(nao_classificados_exemplos) < 10:
                    nao_classificados_exemplos.append(arquivo.name)
                continue

            destino_dir = DESTINO / classe
            destino_dir.mkdir(parents=True, exist_ok=True)

            # prefixa com o split de origem para não perder rastreabilidade
            # nem colidir nome entre splits diferentes
            nome_destino = f"{pasta_origem.name}__{arquivo.name}"
            destino = destino_dir / nome_destino
            if not destino.exists():
                shutil.copy2(arquivo, destino)

            contagem_por_classe[classe] = contagem_por_classe.get(classe, 0) + 1
            total_copiados += 1

    print(f"\n{'=' * 60}\nRESUMO\n{'=' * 60}")
    for classe, n in sorted(contagem_por_classe.items()):
        print(f"  {classe:<10} {n}")
    print(f"\nTotal organizado: {total_copiados}")
    print(f"Não classificados (nome não bateu com nenhuma classe conhecida): {total_nao_classificados}")
    if nao_classificados_exemplos:
        print("Exemplos de arquivo não classificado (confira o padrão de nome):")
        for nome in nao_classificados_exemplos:
            print(f"  {nome}")

    print(f"\nOrganizado em: {DESTINO}")
    print("Próximo passo: rodar montar_dataset_final.py (Fonte 3 já aponta para essa pasta)")


if __name__ == "__main__":
    main()
