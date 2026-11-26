"""
Harmoniza as duas fontes de dataset num dataset unificado.

1. Detecta e corta borda preta (letterbox) automaticamente em QUALQUER
   imagem que tiver — não presume qual fonte tem, mede em cada arquivo.
2. Aplica Resize((256,256)) seguido de CenterCrop(224) em toda imagem —
   mesmo pipeline usado pelo professor (confirmado no notebook do projeto
   de soja), inclusive a distorção de proporção que o Resize com dois
   números causa. De propósito: as duas fontes precisam carregar a mesma
   assinatura de processamento, não a "correta" tecnicamente.
3. Funde V4_V5 -> V4 e V7_V8 -> V7 nas pastas do usuário (Dataset 1),
   unificando com a taxonomia já usada no Dataset 2 (professor).
4. Salva tudo em dataset_unificado/<classe>/, prefixando o nome do
   arquivo com a fonte (usuario_ / professor_) para rastreabilidade.

Uso:
    python harmonizar_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── Ajuste os caminhos abaixo se necessário ────────────────────────────────
PASTA_USUARIO    = Path("Dataset Nova Tentativa")
PASTA_PROFESSOR  = Path("Dataset Nova Tentativa Professor")
DESTINO          = Path("dataset_unificado")

FUSAO_CLASSES_USUARIO = {'V4_V5': 'V4', 'V7_V8': 'V7'}

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp',
              '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP'}
EXTENSIONS_NAO_SUPORTADAS = {'.heic', '.heif', '.HEIC', '.HEIF'}

LIMIAR_PRETO = 15
RESIZE_INTERMEDIARIO = (256, 256)
TAMANHO_FINAL = 224


def medir_bordas(cinza):
    h, w = cinza.shape

    def contar(perfil):
        n = 0
        for v in perfil:
            if v < LIMIAR_PRETO:
                n += 1
            else:
                break
        return n

    media_linhas = cinza.mean(axis=1)
    media_colunas = cinza.mean(axis=0)
    return (contar(media_linhas), contar(media_linhas[::-1]),
            contar(media_colunas), contar(media_colunas[::-1]))


def processar_imagem(path):
    img = Image.open(path).convert('RGB')
    cinza = np.array(img.convert('L'))
    h, w = cinza.shape
    topo, base, esquerda, direita = medir_bordas(cinza)

    tinha_borda = (topo + base + esquerda + direita) > 0
    if (topo + base) < h and (esquerda + direita) < w and tinha_borda:
        img = img.crop((esquerda, topo, w - direita, h - base))

    # Resize((256,256)) direto, sem preservar proporção — de propósito,
    # igual ao pipeline do professor.
    img = img.resize(RESIZE_INTERMEDIARIO, Image.BILINEAR)

    # CenterCrop(224)
    l, a = img.size
    esq = (l - TAMANHO_FINAL) // 2
    top = (a - TAMANHO_FINAL) // 2
    img = img.crop((esq, top, esq + TAMANHO_FINAL, top + TAMANHO_FINAL))

    return img, tinha_borda


def processar_fonte(pasta_origem, prefixo_fonte, mapa_classes=None):
    if not pasta_origem.exists():
        print(f"[ERRO] Pasta não encontrada: {pasta_origem}")
        return

    resumo = {}
    for cls_dir in sorted(p for p in pasta_origem.iterdir() if p.is_dir()):
        classe_original = cls_dir.name
        classe_destino = (mapa_classes or {}).get(classe_original, classe_original)

        destino_dir = DESTINO / classe_destino
        destino_dir.mkdir(parents=True, exist_ok=True)

        n_ok, n_borda, n_erro, n_nao_suportado = 0, 0, 0, 0

        for arquivo in cls_dir.iterdir():
            if arquivo.suffix in EXTENSIONS_NAO_SUPORTADAS:
                n_nao_suportado += 1
                continue
            if arquivo.suffix not in EXTENSIONS:
                continue

            nome_destino = f"{prefixo_fonte}_{arquivo.stem}.jpg"
            destino = destino_dir / nome_destino
            if destino.exists():
                n_ok += 1
                continue

            try:
                img_final, tinha_borda = processar_imagem(arquivo)
                img_final.save(destino, quality=95)
                n_ok += 1
                if tinha_borda:
                    n_borda += 1
            except Exception as e:
                print(f"    [ERRO] {arquivo.name}: {e}")
                n_erro += 1

        classe_label = classe_original if classe_original == classe_destino else f"{classe_original} -> {classe_destino}"
        print(f"  {classe_label:<15} {n_ok:>5} processadas | {n_borda:>4} tinham borda | "
              f"{n_erro} erro(s) | {n_nao_suportado} não suportada(s) (HEIC/HEIF)")

        chave = classe_destino
        if chave not in resumo:
            resumo[chave] = {'ok': 0, 'borda': 0}
        resumo[chave]['ok'] += n_ok
        resumo[chave]['borda'] += n_borda

    return resumo


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)

    print(f"Processando Dataset 1 (usuário): {PASTA_USUARIO}")
    resumo_usuario = processar_fonte(PASTA_USUARIO, "usuario", FUSAO_CLASSES_USUARIO)

    print(f"\nProcessando Dataset 2 (professor): {PASTA_PROFESSOR}")
    resumo_professor = processar_fonte(PASTA_PROFESSOR, "professor")

    print(f"\n{'=' * 60}\nRESUMO — borda detectada por fonte\n{'=' * 60}")
    total_borda_usuario = sum(r['borda'] for r in (resumo_usuario or {}).values())
    total_ok_usuario = sum(r['ok'] for r in (resumo_usuario or {}).values())
    total_borda_prof = sum(r['borda'] for r in (resumo_professor or {}).values())
    total_ok_prof = sum(r['ok'] for r in (resumo_professor or {}).values())

    print(f"Usuário  : {total_borda_usuario}/{total_ok_usuario} imagens tinham borda "
          f"({100 * total_borda_usuario / max(total_ok_usuario, 1):.1f}%)")
    print(f"Professor: {total_borda_prof}/{total_ok_prof} imagens tinham borda "
          f"({100 * total_borda_prof / max(total_ok_prof, 1):.1f}%)")

    print(f"\nDataset unificado salvo em: {DESTINO}")


if __name__ == "__main__":
    main()
