"""
Monta o dataset unificado a partir de 3 fontes, cada uma com o tratamento
que ela precisa:

  Fonte 1 (usuário, S24 Ultra)  -> corta borda (não deveria ter, mas mede
                                    de qualquer forma) + harmoniza resolução
  Fonte 2 (professor, 224x224)  -> corta borda (100% das imagens tinham) +
                                    harmoniza resolução
  Fonte 3 (externo V1-V4, ~15k) -> corta borda + harmoniza resolução +
                                    DEDUPLICAÇÃO POR HASH DE CONTEÚDO
                                    (obrigatória — esse banco tem 41-66%
                                    de duplicata exata confirmada, com
                                    nomes de arquivo diferentes, o que
                                    escapa de qualquer agrupamento por
                                    nome de arquivo)

Harmonização = Resize((256,256)) [distorce proporção, de propósito, igual
ao pipeline do professor] + CenterCrop(224).

Dedup = MD5 exato (remove) + average hash perceptual (agrupa quase-
duplicata via Union-Find, não remove — só marca para nunca serem
separadas entre treino/val/teste depois, no K-Fold).

Os caminhos das 3 fontes ficam na seção CONFIGURAÇÃO abaixo. O script
pula a fonte cuja pasta não encontrar.

Uso:
    python montar_dataset_final.py
"""

import sys
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── CONFIGURAÇÃO ───────────────────────────────────────────────────────────
FONTES = [
    {
        'pasta': Path("Dataset Nova Tentativa"),
        'prefixo': 'usuario',
        'mapa_classes': {'V4_V5': 'V4', 'V7_V8': 'V7'},
        'deduplicar': False,  # fotos próprias, sem motivo pra duplicata
    },
    {
        'pasta': Path("Dataset Nova Tentativa Professor"),
        'prefixo': 'professor',
        'mapa_classes': None,
        'deduplicar': False,  # já é o dataset dele, tratado à parte
    },
    {
        'pasta': Path("Dataset V1 a V4 Externo"),
        'prefixo': 'externo',
        'mapa_classes': None,
        'deduplicar': True,  # OBRIGATÓRIO — banco com duplicata confirmada
    },
]

DESTINO = Path("dataset_unificado")
GRUPOS_SAIDA = Path("grupos_quase_duplicata.json")  # para o K-Fold usar depois

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp',
              '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP'}
EXTENSIONS_NAO_SUPORTADAS = {'.heic', '.heif', '.HEIC', '.HEIF'}

LIMIAR_PRETO = 15
LIMIAR_QUASE_DUPLICATA = 5  # distância Hamming, average hash 16x16 (256 bits)
RESIZE_INTERMEDIARIO = (256, 256)
TAMANHO_FINAL = 224


# ─── Borda + harmonização de resolução ──────────────────────────────────────

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

    img = img.resize(RESIZE_INTERMEDIARIO, Image.BILINEAR)

    l, a = img.size
    esq = (l - TAMANHO_FINAL) // 2
    top = (a - TAMANHO_FINAL) // 2
    img = img.crop((esq, top, esq + TAMANHO_FINAL, top + TAMANHO_FINAL))

    return img, tinha_borda


# ─── Deduplicação por hash de conteúdo ──────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self.pai = list(range(n))

    def find(self, x):
        while self.pai[x] != x:
            self.pai[x] = self.pai[self.pai[x]]
            x = self.pai[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.pai[rx] = ry


def deduplicar_lista(itens):
    """itens: lista de (path_original, imagem_ja_harmonizada_PIL). Retorna
    lista filtrada (sem duplicata exata) + dict path->grupo_id."""
    n = len(itens)
    hashes_md5 = []
    hashes_ahash = []
    for _, img in itens:
        buf = np.array(img.convert('RGB'))
        hashes_md5.append(hashlib.md5(buf.tobytes()).hexdigest())
        cinza16 = np.array(img.convert('L').resize((16, 16)), dtype=np.float64)
        hashes_ahash.append((cinza16 > cinza16.mean()).flatten())

    uf = UnionFind(n)
    md5_idx = defaultdict(list)
    for i, h in enumerate(hashes_md5):
        md5_idx[h].append(i)
    for idxs in md5_idx.values():
        for i in idxs[1:]:
            uf.union(idxs[0], i)

    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            d = np.count_nonzero(hashes_ahash[i] != hashes_ahash[j])
            if d <= LIMIAR_QUASE_DUPLICATA:
                uf.union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    mantidos = []
    grupo_por_path = {}
    for cluster_idxs in clusters.values():
        vistos_md5 = set()
        cluster_id = f"cluster{min(cluster_idxs)}"
        for i in cluster_idxs:
            path, img = itens[i]
            if hashes_md5[i] in vistos_md5:
                continue  # duplicata exata dentro do cluster — descarta
            vistos_md5.add(hashes_md5[i])
            mantidos.append((path, img))
            grupo_por_path[str(path)] = cluster_id

    return mantidos, grupo_por_path


# ─── Processamento por fonte ─────────────────────────────────────────────────

def processar_fonte(fonte, pool_por_classe):
    pasta = fonte['pasta']
    if not pasta.exists():
        print(f"[AVISO] Pasta não encontrada, pulando fonte '{fonte['prefixo']}': {pasta}")
        return

    mapa = fonte['mapa_classes'] or {}
    for cls_dir in sorted(p for p in pasta.iterdir() if p.is_dir()):
        classe_destino = mapa.get(cls_dir.name, cls_dir.name)
        n_ok, n_borda, n_erro, n_nao_suportado = 0, 0, 0, 0

        for arquivo in cls_dir.iterdir():
            if arquivo.suffix in EXTENSIONS_NAO_SUPORTADAS:
                n_nao_suportado += 1
                continue
            if arquivo.suffix not in EXTENSIONS:
                continue
            try:
                img_final, tinha_borda = processar_imagem(arquivo)
                nome_destino = f"{fonte['prefixo']}_{arquivo.stem}"
                pool_por_classe[classe_destino].append((nome_destino, img_final))
                n_ok += 1
                if tinha_borda:
                    n_borda += 1
            except Exception as e:
                print(f"    [ERRO] {arquivo.name}: {e}")
                n_erro += 1

        label = cls_dir.name if cls_dir.name == classe_destino else f"{cls_dir.name} -> {classe_destino}"
        print(f"  [{fonte['prefixo']}] {label:<15} {n_ok:>5} ok | {n_borda:>4} com borda | "
              f"{n_erro} erro(s) | {n_nao_suportado} não suportada(s)")


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    pool_por_classe = defaultdict(list)  # classe -> [(nome_base, PIL.Image), ...]
    fontes_com_dedup = set()

    for fonte in FONTES:
        print(f"\nProcessando fonte '{fonte['prefixo']}': {fonte['pasta']}")
        processar_fonte(fonte, pool_por_classe)
        if fonte['deduplicar']:
            fontes_com_dedup.add(fonte['prefixo'])

    print(f"\n{'=' * 60}\nDeduplicando e salvando\n{'=' * 60}")
    grupos_finais = {}

    for classe, itens in sorted(pool_por_classe.items()):
        precisa_dedup = any(nome.startswith(tuple(f"{p}_" for p in fontes_com_dedup)) for nome, _ in itens)

        if precisa_dedup:
            antes = len(itens)
            mantidos, grupos = deduplicar_lista(itens)
            removidos = antes - len(mantidos)
            print(f"  {classe:<10} {antes:>5} -> {len(mantidos):>5} ({removidos} duplicata(s) removida(s))")
        else:
            mantidos = itens
            grupos = {nome: f"unico_{nome}" for nome, _ in itens}
            print(f"  {classe:<10} {len(itens):>5} (sem dedup — fonte não marcada como duplicável)")

        destino_dir = DESTINO / classe
        destino_dir.mkdir(parents=True, exist_ok=True)
        for nome_base, img in mantidos:
            destino = destino_dir / f"{nome_base}.jpg"
            if not destino.exists():
                img.save(destino, quality=95)
            grupos_finais[str(destino)] = grupos.get(nome_base, f"unico_{nome_base}")

    with open(GRUPOS_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(grupos_finais, f, indent=2, ensure_ascii=False)

    print(f"\nDataset final em: {DESTINO}")
    print(f"Grupos de quase-duplicata salvos em: {GRUPOS_SAIDA} "
          f"(usar como 'origem' no K-Fold, não o nome do arquivo)")


if __name__ == "__main__":
    main()
