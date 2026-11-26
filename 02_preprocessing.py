"""
Preprocessing: undersampling, holdout, data augmentation e K-Fold.

Balanceia cada classe de CLASSES_TREINO em torno de TARGET_N_AUG, separa um
holdout fixo antes de qualquer augmentation, gera as imagens aumentadas
rastreando a imagem de origem de cada cópia, e grava a configuração do
K-Fold usando StratifiedGroupKFold — garantindo que cópias aumentadas da
mesma foto nunca fiquem separadas entre treino e validação no mesmo fold.

Remove execução anterior (augmented/, test/, kfold_config.json) antes de
começar, para que reruns com parâmetros diferentes não misturem dado velho
com novo.

Uso:
    python 02_preprocessing.py
"""

import sys
import json
import random
import shutil
from pathlib import Path

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageFile
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from config import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp',
              '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP'}
LIMIAR_QUASE_DUPLICATA = 5  # distância Hamming, average hash 16x16 (256 bits)


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


def calcular_hashes(paths):
    registros = []
    for p in paths:
        try:
            with open(p, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            img = Image.open(p).convert('L').resize((16, 16))
            arr = np.array(img, dtype=np.float64)
            ahash = (arr > arr.mean()).flatten()
            registros.append({'path': p, 'md5': md5, 'ahash': ahash})
        except Exception:
            continue
    return registros


def deduplicar_e_agrupar(registros):
    n = len(registros)
    uf = UnionFind(n)

    md5_idx = defaultdict(list)
    for i, r in enumerate(registros):
        md5_idx[r['md5']].append(i)
    for idxs in md5_idx.values():
        for i in idxs[1:]:
            uf.union(idxs[0], i)

    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            d = np.count_nonzero(registros[i]['ahash'] != registros[j]['ahash'])
            if d <= LIMIAR_QUASE_DUPLICATA:
                uf.union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    mantidos = []
    for cluster_idxs in clusters.values():
        vistos_md5 = set()
        cluster_id = f"cluster{min(cluster_idxs)}"
        for i in cluster_idxs:
            r = registros[i]
            if r['md5'] in vistos_md5:
                continue  # duplicata exata dentro do cluster — descarta
            vistos_md5.add(r['md5'])
            mantidos.append({'path': str(r['path']), 'origem': cluster_id})

    return mantidos


def limpar_execucao_anterior():
    for path in [AUGMENTED_DIR, SPLIT_DIR / 'test']:
        if path.exists():
            shutil.rmtree(path)
    for arquivo in [KFOLD_CONFIG, RESULT_DIR / 'excedente.json']:
        if arquivo.exists():
            arquivo.unlink()
    print("Execução anterior removida (augmented/, test/, kfold_config.json, excedente.json).\n")


def inventariar():
    registros = []
    for cls in CLASSES_TREINO:
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            continue

        paths_validos = []
        for img_path in cls_dir.iterdir():
            if img_path.suffix not in EXTENSIONS:
                continue
            if img_path.name in ARQUIVOS_EXCLUIDOS:
                continue
            try:
                with Image.open(img_path) as im:
                    im.verify()
                paths_validos.append(img_path)
            except Exception:
                continue

        print(f"  {cls}: deduplicando {len(paths_validos)} imagens (hash de conteúdo)...")
        hashes = calcular_hashes(paths_validos)
        mantidos = deduplicar_e_agrupar(hashes)
        removidos = len(paths_validos) - len(mantidos)
        print(f"    {len(mantidos)} mantidas, {removidos} duplicata(s) exata(s) removida(s)")

        for m in mantidos:
            registros.append({'path': m['path'], 'classe': cls, 'origem': f"{cls}_{m['origem']}"})

    return pd.DataFrame(registros)


def undersample(df):
    random.seed(SEED)
    partes = []
    excedente = {}
    print(f"Aplicando undersampling (TETO_MAX={TETO_MAX})...\n")
    for cls in CLASSES_TREINO:
        subset = df[df['classe'] == cls]
        n = len(subset)
        if n > TETO_MAX:
            mantidas = subset.sample(n=TETO_MAX, random_state=SEED)
            descartadas = subset.drop(mantidas.index)
            excedente[cls] = descartadas['path'].tolist()
            print(f"  {cls:<8}: {n} → {TETO_MAX} (reservadas {len(descartadas)} para excedente.json)")
            partes.append(mantidas)
        else:
            print(f"  {cls:<8}: {n} (mantidas)")
            partes.append(subset)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / 'excedente.json', 'w', encoding='utf-8') as f:
        json.dump(excedente, f, indent=2, ensure_ascii=False)
    print(f"\nExcedente registrado em: {RESULT_DIR / 'excedente.json'} "
          f"({sum(len(v) for v in excedente.values())} imagens reservadas, não descartadas)")

    return pd.concat(partes).reset_index(drop=True)


def separar_holdout(df):
    df_trainval, df_test = train_test_split(
        df, test_size=TEST_RATIO, stratify=df['classe'], random_state=SEED,
    )
    print(f"\nTrain+Val : {len(df_trainval)} imagens")
    print(f"Test      : {len(df_test)} imagens (holdout fixo)")
    print(df_test['classe'].value_counts().reindex(CLASSES_TREINO))

    test_dir = SPLIT_DIR / 'test'
    for cls in CLASSES_TREINO:
        (test_dir / cls).mkdir(parents=True, exist_ok=True)

    copiadas = 0
    for _, row in df_test.iterrows():
        src = Path(row['path'])
        dst = test_dir / row['classe'] / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copiadas += 1
    print(f"\n{copiadas} imagens copiadas para test/")
    return df_trainval, df_test


def variar_hsv(img, graus_h, fator_s, fator_v):
    """Varia matiz, saturação e brilho juntos. Só matiz não bastava —
    diagnóstico controlado mostrou que a marca de fonte estava mais em
    saturação/brilho (queda de 0,86->0,63 com os três canais, contra
    queda de só 0,01 variando só matiz)."""
    deslocamento = int(round(graus_h / 360 * 256))
    hsv = img.convert('HSV')
    h, s, v = hsv.split()
    h_arr = (np.array(h, dtype=np.int16) + deslocamento) % 256
    s_arr = np.clip(np.array(s, dtype=np.float64) * fator_s, 0, 255)
    v_arr = np.clip(np.array(v, dtype=np.float64) * fator_v, 0, 255)
    h_novo = Image.fromarray(h_arr.astype(np.uint8), mode='L')
    s_novo = Image.fromarray(s_arr.astype(np.uint8), mode='L')
    v_novo = Image.fromarray(v_arr.astype(np.uint8), mode='L')
    return Image.merge('HSV', (h_novo, s_novo, v_novo)).convert('RGB')


def augmentar_imagem(img, seed):
    rng = random.Random(seed)

    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.3:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))

    img = variar_hsv(img, rng.uniform(-18, 18), rng.uniform(0.6, 1.4), rng.uniform(0.6, 1.4))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.6))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.4))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.5))
    img = ImageEnhance.Sharpness(img).enhance(rng.uniform(0.5, 2.0))

    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

    if rng.random() < 0.4:
        w, h = img.size
        left   = int(rng.uniform(0, 0.15) * w)
        top    = int(rng.uniform(0, 0.15) * h)
        right  = int((1 - rng.uniform(0, 0.15)) * w)
        bottom = int((1 - rng.uniform(0, 0.15)) * h)
        img = img.crop((left, top, right, bottom))

    return img


def gerar_augmentation(df_trainval):
    print("\nAplicando Data Augmentation...\n")
    registros = []

    for cls in CLASSES_TREINO:
        cls_dir = AUGMENTED_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        subset = df_trainval[df_trainval['classe'] == cls]
        n_orig = len(subset)
        n_needed = max(0, TARGET_N_AUG - n_orig)
        n_per_img = int(np.ceil(n_needed / n_orig)) if n_orig > 0 else 0

        for _, row in subset.iterrows():
            src = Path(row['path'])
            dst = cls_dir / src.name
            if not dst.exists():
                try:
                    img_src = Image.open(src).convert('RGB')
                    seed_local = (SEED + abs(hash((cls, src.name)))) % 1000000
                    rng_local = random.Random(seed_local)
                    graus = rng_local.uniform(-18, 18)
                    fator_s = rng_local.uniform(0.6, 1.4)
                    fator_v = rng_local.uniform(0.6, 1.4)
                    img_jitter = variar_hsv(img_src, graus, fator_s, fator_v)
                    img_jitter.save(dst, quality=95)
                    with Image.open(dst) as im:
                        im.verify()
                except Exception as e:
                    print(f"  [AVISO] {dst.name} copiado mas ilegível logo em seguida — "
                          f"provável antivírus removendo/quarentenando o arquivo. Pulando. ({e})")
                    dst.unlink(missing_ok=True)
                    continue
            registros.append({'path': str(dst), 'classe': cls, 'origem': row['origem']})

        gerados = 0
        origens = subset[['path', 'origem']].to_dict('records') * (n_per_img + 1)
        for i, origem_row in enumerate(origens):
            if gerados >= n_needed:
                break
            try:
                origem = origem_row['origem']
                img = Image.open(origem_row['path']).convert('RGB')
                aug = augmentar_imagem(img, seed=SEED + i + hash(cls) % 10000)
                nome = f"aug_{origem}_{gerados:03d}.jpg"
                dst = cls_dir / nome
                aug.save(dst, quality=92)
                registros.append({'path': str(dst), 'classe': cls, 'origem': origem})
                gerados += 1
            except Exception as e:
                print(f"  Erro em {origem_row['path']}: {e}")

        total = n_orig + gerados
        print(f"  {cls:<8}: {n_orig:>3} orig + {gerados:>3} geradas = {total:>3} total")

    return pd.DataFrame(registros)


def montar_kfold(df_aug, df_test):
    df_aug['label'] = df_aug['classe'].map(LABEL_TO_IDX)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []

    print(f"\nDataset augmentado: {len(df_aug)} imagens ({df_aug['origem'].nunique()} origens únicas)")
    for fold_idx, (train_idx, val_idx) in enumerate(
            sgkf.split(df_aug['path'], df_aug['label'], groups=df_aug['origem'])):
        train_origens = set(df_aug.iloc[train_idx]['origem'])
        val_origens = set(df_aug.iloc[val_idx]['origem'])
        assert not (train_origens & val_origens), f"Vazamento de origem no fold {fold_idx}"

        folds.append({
            'fold': fold_idx,
            'train_paths': df_aug.iloc[train_idx]['path'].tolist(),
            'train_labels': df_aug.iloc[train_idx]['label'].tolist(),
            'val_paths': df_aug.iloc[val_idx]['path'].tolist(),
            'val_labels': df_aug.iloc[val_idx]['label'].tolist(),
        })
        print(f"  Fold {fold_idx + 1}: {len(train_idx)} treino | {len(val_idx)} val — sem sobreposição de origem")

    kfold_meta = {
        'estrategia': 'StratifiedGroupKFold',
        'n_folds': N_FOLDS,
        'seed': SEED,
        'label_to_idx': LABEL_TO_IDX,
        'idx_to_label': IDX_TO_LABEL,
        'classes': CLASSES_TREINO,
        'excluded_classes': EXCLUDED_CLASSES,
        'total_aug': len(df_aug),
        'folds': folds,
        'test_paths': df_test['path'].tolist(),
        'test_labels': df_test['classe'].map(LABEL_TO_IDX).tolist(),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(KFOLD_CONFIG, 'w', encoding='utf-8') as f:
        json.dump(kfold_meta, f, indent=2, ensure_ascii=False)

    print(f"\nK-Fold config salvo em: {KFOLD_CONFIG}")


def main():
    limpar_execucao_anterior()
    df = inventariar()
    df = undersample(df)
    df_trainval, df_test = separar_holdout(df)
    df_aug = gerar_augmentation(df_trainval)
    montar_kfold(df_aug, df_test)


if __name__ == "__main__":
    main()
