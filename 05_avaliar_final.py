"""
Avaliação final consolidada.

resultado_<modelo>.json só tem F1-macro agregado dos folds — não quebra
por classe. A H1 do TCC compara arquiteturas especificamente nos estádios
vegetativos vs. reprodutivos, então isso precisa de F1 por classe, não só
o agregado.

Para cada modelo, faz ensemble dos 5 folds (média das probabilidades
softmax) sobre o conjunto de teste fixo, e gera:
  - outputs/results/avaliacao_<modelo>.json  (classification_report completo
    por classe + matriz de confusão)
  - outputs/results/comparativo_final.csv    (F1-macro e accuracy dos 5
    modelos, um do lado do outro)

Uso:
    python 05_avaliar_final.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import timm

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from treino_core import MilhoDataset, build_transform

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def carregar_ensemble(modelo, cfg):
    modelos = []
    for fold_idx in range(N_FOLDS):
        path = MODEL_DIR / f"{modelo}_fold{fold_idx + 1}.pt"
        if not path.exists():
            print(f"  [AVISO] {path.name} não encontrado — pulando esse fold no ensemble")
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
            probs_batch = torch.zeros(imgs.size(0), NUM_CLASSES, device=DEVICE)
            for m in modelos:
                probs_batch += Fn.softmax(m(imgs), dim=1)
            probs_batch /= len(modelos)
        todas_probs.append(probs_batch.cpu())
        todas_labels.append(labels)

    probs = torch.cat(todas_probs).numpy()
    labels = torch.cat(todas_labels).numpy()
    return probs.argmax(axis=1), labels


def main():
    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)

    nomes_classes = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    resumo_geral = []

    for modelo, cfg in MODELS.items():
        resultado_path = RESULT_DIR / f"resultado_{modelo}.json"
        if not resultado_path.exists():
            print(f"{modelo}: sem resultado_{modelo}.json — treino não concluído, pulando")
            continue

        print(f"\n{'=' * 60}\n{modelo}\n{'=' * 60}")
        modelos = carregar_ensemble(modelo, cfg)
        if not modelos:
            print("  Nenhum fold encontrado, pulando")
            continue

        transform = build_transform(cfg['img_size'])
        test_ds = MilhoDataset(kfold['test_paths'], kfold['test_labels'], transform)
        test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=NUM_WORKERS)

        preds, labels = prever_ensemble(modelos, test_loader)

        relatorio = classification_report(labels, preds, target_names=nomes_classes,
                                           output_dict=True, zero_division=0)
        matriz = confusion_matrix(labels, preds)

        print(classification_report(labels, preds, target_names=nomes_classes, zero_division=0))

        with open(RESULT_DIR / f"avaliacao_{modelo}.json", 'w', encoding='utf-8') as f:
            json.dump({
                'modelo': modelo,
                'classification_report': relatorio,
                'confusion_matrix': matriz.tolist(),
                'classes': nomes_classes,
            }, f, indent=2, ensure_ascii=False)

        resumo_geral.append({
            'modelo': modelo,
            'f1_macro_ensemble_teste': relatorio['macro avg']['f1-score'],
            'accuracy_teste': relatorio['accuracy'],
        })

        for m in modelos:
            del m
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    if not resumo_geral:
        print("\nNenhum modelo com resultado encontrado.")
        return

    print(f"\n{'=' * 60}\nRESUMO COMPARATIVO (ensemble dos 5 folds, conjunto de teste)\n{'=' * 60}")
    df = pd.DataFrame(resumo_geral).sort_values('f1_macro_ensemble_teste', ascending=False)
    print(df.to_string(index=False))

    df.to_csv(RESULT_DIR / 'comparativo_final.csv', index=False)
    print(f"\nSalvo em: {RESULT_DIR / 'comparativo_final.csv'}")
    print(f"F1 por classe e matriz de confusão de cada modelo em: {RESULT_DIR}\\avaliacao_<modelo>.json")


if __name__ == "__main__":
    main()
