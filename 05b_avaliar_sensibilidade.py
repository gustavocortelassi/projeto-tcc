"""
Análise de sensibilidade do conjunto de teste ao vazamento por quase-duplicata.

O holdout do 02_preprocessing.py foi estratificado por classe, mas não por
grupo; parte das imagens de teste tem quase-duplicata (rajada de fotos) no
treino+val. Este script mede o efeito disso SEM retreinar:

  1. Identifica as imagens de teste com vizinho no treino+val pelo mesmo
     critério do pipeline (average hash 16x16, Hamming <= 5) e por um
     critério folgado (<= 10).
  2. Roda o ensemble dos 5 folds (média das softmax, igual ao 05) UMA vez.
  3. Calcula métricas em três versões do teste: completo, limpo_5, limpo_10.
  4. Separa F1 macro por grupo: vegetativos por colar (V1..V12),
     reprodutivos (R*) e VT.

Não sobrescreve nenhum arquivo existente. Saídas novas:
  outputs/results/teste_vizinhos_treino.json
  outputs/results/sensibilidade_<modelo>.json
  outputs/results/sensibilidade_comparativo.csv

Uso:
    python 05b_avaliar_sensibilidade.py
"""

import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
import timm

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from treino_core import MilhoDataset, build_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
LIMIARES = {'limpo_5': 5, 'limpo_10': 10}


# ─── Grupos de estádio ───────────────────────────────────────────────────────
def grupo_de(classe):
    if re.fullmatch(r"V\d+", classe):
        return 'vegetativo_colar'
    if classe.startswith('R'):
        return 'reprodutivo'
    return classe  # VT


# ─── 1. Vizinhos no treino ───────────────────────────────────────────────────
def ahash(path):
    img = Image.open(path).convert('L').resize((16, 16))
    arr = np.array(img, dtype=np.float64)
    return (arr > arr.mean()).flatten()


def originais_trainval(cls):
    pasta = AUGMENTED_DIR / cls
    if not pasta.exists():
        return []
    return [RAW_DIR / cls / f.name for f in pasta.iterdir()
            if f.is_file() and f.suffix.lower() in EXT and not f.name.startswith("aug_")]


def distancia_minima_ao_treino(test_paths):
    """Retorna dict path_teste -> (distância mínima, vizinho mais próximo)."""
    por_classe = {}
    for p in test_paths:
        por_classe.setdefault(Path(p).parent.name, []).append(p)

    resultado = {}
    for cls, paths in por_classe.items():
        treino = [p for p in originais_trainval(cls) if p.exists()]
        if not treino:
            for p in paths:
                resultado[p] = (999, None)
            continue
        H_te = np.array([ahash(p) for p in paths])
        H_tr = np.array([ahash(p) for p in treino])
        dist = (H_te[:, None, :] != H_tr[None, :, :]).sum(axis=2)
        for i, p in enumerate(paths):
            j = int(dist[i].argmin())
            resultado[p] = (int(dist[i, j]), treino[j].name)
    return resultado


# ─── 2. Ensemble (mesma lógica do 05_avaliar_final.py) ──────────────────────
def carregar_ensemble(modelo, cfg):
    modelos = []
    for fold_idx in range(N_FOLDS):
        path = MODEL_DIR / f"{modelo}_fold{fold_idx + 1}.pt"
        if not path.exists():
            print(f"  [AVISO] {path.name} não encontrado — fold fora do ensemble")
            continue
        m = timm.create_model(cfg['timm_name'], pretrained=False, num_classes=NUM_CLASSES)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m.to(DEVICE).eval()
        modelos.append(m)
    return modelos


def prever_ensemble(modelos, loader):
    todas_probs, todas_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
            probs = torch.zeros(imgs.size(0), NUM_CLASSES, device=DEVICE)
            for m in modelos:
                probs += Fn.softmax(m(imgs), dim=1)
            probs /= len(modelos)
        todas_probs.append(probs.float().cpu())
        todas_labels.append(labels)
    return torch.cat(todas_probs).numpy().argmax(axis=1), torch.cat(todas_labels).numpy()


# ─── 3. Métricas por versão do teste ─────────────────────────────────────────
def metricas(labels, preds, nomes):
    rel = classification_report(labels, preds, labels=list(range(len(nomes))),
                                target_names=nomes, output_dict=True, zero_division=0)
    f1_classe = {c: rel[c]['f1-score'] for c in nomes}
    suporte = {c: int(rel[c]['support']) for c in nomes}

    por_grupo = {}
    for g in ('vegetativo_colar', 'reprodutivo', 'VT'):
        cls_g = [c for c in nomes if grupo_de(c) == g and suporte[c] > 0]
        por_grupo[g] = float(np.mean([f1_classe[c] for c in cls_g])) if cls_g else None

    presentes = [i for i, c in enumerate(nomes) if suporte[c] > 0]
    return {
        'n': int(len(labels)),
        'erros': int((labels != preds).sum()),
        'accuracy': float(accuracy_score(labels, preds)),
        'f1_macro': float(f1_score(labels, preds, labels=presentes, average='macro', zero_division=0)),
        'f1_macro_por_grupo': por_grupo,
        'f1_por_classe': f1_classe,
        'suporte_por_classe': suporte,
    }


def main():
    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)
    test_paths = kfold['test_paths']
    test_labels = np.array(kfold['test_labels'])
    nomes = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

    # 1. vizinhos
    print("Calculando distância de cada imagem de teste ao treino+val...")
    viz = distancia_minima_ao_treino(test_paths)
    dmin = np.array([viz[p][0] for p in test_paths])
    mascaras = {'completo': np.ones(len(test_paths), dtype=bool)}
    for nome, lim in LIMIARES.items():
        mascaras[nome] = dmin > lim
        print(f"  {nome}: {int(mascaras[nome].sum())} de {len(test_paths)} imagens mantidas")

    with open(RESULT_DIR / 'teste_vizinhos_treino.json', 'w', encoding='utf-8') as f:
        json.dump([{'teste': p, 'classe': Path(p).parent.name,
                    'dist_min': viz[p][0], 'vizinho_treino': viz[p][1]}
                   for p in test_paths], f, indent=2, ensure_ascii=False)

    # 2-3. modelos
    linhas = []
    for modelo, cfg in MODELS.items():
        print(f"\n{'=' * 60}\n{modelo}\n{'=' * 60}")
        modelos = carregar_ensemble(modelo, cfg)
        if not modelos:
            print("  Nenhum fold encontrado, pulando")
            continue

        ds = MilhoDataset(test_paths, test_labels.tolist(), build_transform(cfg['img_size']))
        loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=NUM_WORKERS)
        preds, labels = prever_ensemble(modelos, loader)
        assert (labels == test_labels).all(), "Ordem do DataLoader diverge do kfold_config"

        saida = {'modelo': modelo, 'n_folds_ensemble': len(modelos), 'versoes': {}}
        linha = {'modelo': modelo}
        for versao, mask in mascaras.items():
            m = metricas(labels[mask], preds[mask], nomes)
            saida['versoes'][versao] = m
            linha[f'f1_{versao}'] = m['f1_macro']
            linha[f'acc_{versao}'] = m['accuracy']
            linha[f'erros_{versao}'] = m['erros']
            linha[f'n_{versao}'] = m['n']
            for g, v in m['f1_macro_por_grupo'].items():
                linha[f'f1_{g}_{versao}'] = v
            print(f"  {versao:9} n={m['n']:>3}  F1={m['f1_macro']:.4f}  acc={m['accuracy']:.4f}  "
                  f"erros={m['erros']:>2}  | veg={m['f1_macro_por_grupo']['vegetativo_colar']:.4f}  "
                  f"rep={m['f1_macro_por_grupo']['reprodutivo']:.4f}  VT={m['f1_macro_por_grupo']['VT']:.4f}")

        with open(RESULT_DIR / f'sensibilidade_{modelo}.json', 'w', encoding='utf-8') as f:
            json.dump(saida, f, indent=2, ensure_ascii=False)
        linhas.append(linha)

        del modelos
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    if not linhas:
        print("\nNenhum modelo avaliado.")
        return

    df = pd.DataFrame(linhas).sort_values('f1_completo', ascending=False)
    df.to_csv(RESULT_DIR / 'sensibilidade_comparativo.csv', index=False)

    print(f"\n{'=' * 60}\nRESUMO (F1 macro)\n{'=' * 60}")
    cols = ['modelo', 'f1_completo', 'f1_limpo_5', 'f1_limpo_10',
            'f1_vegetativo_colar_completo', 'f1_vegetativo_colar_limpo_5']
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSalvo em: {RESULT_DIR / 'sensibilidade_comparativo.csv'}")


if __name__ == "__main__":
    main()
