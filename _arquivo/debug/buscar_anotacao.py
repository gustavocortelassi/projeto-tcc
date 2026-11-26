"""
Busca qualquer arquivo que não seja imagem dentro de train_set/,
validation_set/, test_set/ — candidato a arquivo de anotação (bounding
box + classe por objeto), que é o que falta para rotular os frames brutos
(IMG_1822_01.JPG etc., sem classe no nome).

Uso:
    python buscar_anotacao.py
"""

from pathlib import Path

PASTAS = [Path("train_set"), Path("validation_set"), Path("test_set")]
EXTENSIONS_IMAGEM = {'.jpg', '.jpeg', '.png', '.bmp', '.webp',
                      '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP'}


def main():
    encontrados = []
    for pasta in PASTAS:
        if not pasta.exists():
            continue
        for arquivo in pasta.rglob('*'):
            if arquivo.is_file() and arquivo.suffix not in EXTENSIONS_IMAGEM:
                encontrados.append(arquivo)

    if not encontrados:
        print("Nenhum arquivo de anotação encontrado — só imagem pura nas três pastas.")
        return

    print(f"{len(encontrados)} arquivo(s) não-imagem encontrado(s):\n")
    for f in encontrados[:30]:
        tamanho_kb = f.stat().st_size / 1024
        print(f"  {f}  ({tamanho_kb:.1f} KB)")
    if len(encontrados) > 30:
        print(f"  ... e mais {len(encontrados) - 30}")


if __name__ == "__main__":
    main()
