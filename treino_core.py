"""
Lógica de treino K-Fold compartilhada por todos os modelos do benchmark.

Não roda diretamente — é importado por 03_treinar.py (CLI com --modelo) e
pelos treinar_<modelo>.py individuais. Toda a lógica de checkpoint por
época, retomada automática e resultado incremental por fold vive aqui, uma
vez só, para não divergir entre os 5 modelos.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
import timm
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

sys.path.insert(0, str(Path(__file__).parent))
from config import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MilhoDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception as e:
            print(f"  [AVISO] Falha ao ler {self.paths[idx]}: {e} — usando imagem seguinte no lugar")
            idx = (idx + 1) % len(self.paths)
            img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def checkpoint_path(modelo, fold_idx):
    return MODEL_DIR / f"{modelo}_fold{fold_idx + 1}_checkpoint.pt"


def final_model_path(modelo, fold_idx):
    return MODEL_DIR / f"{modelo}_fold{fold_idx + 1}.pt"


def fold_resultado_path(modelo, fold_idx):
    return RESULT_DIR / f"{modelo}_fold{fold_idx + 1}_resultado.json"


def resultado_path(modelo):
    return RESULT_DIR / f"resultado_{modelo}.json"


def validar_kfold(kfold):
    faltando = []
    for fold in kfold['folds']:
        faltando += [p for p in fold['train_paths'] if not Path(p).exists()]
        faltando += [p for p in fold['val_paths'] if not Path(p).exists()]
    faltando += [p for p in kfold['test_paths'] if not Path(p).exists()]

    total = sum(len(f['train_paths']) + len(f['val_paths']) for f in kfold['folds']) + len(kfold['test_paths'])

    if faltando:
        print(f"\n[ERRO] {len(faltando)} de {total} arquivo(s) do kfold_config.json não existem mais em disco:")
        for p in faltando[:10]:
            print(f"  {p}")
        if len(faltando) > 10:
            print(f"  ... e mais {len(faltando) - 10}")
        raise FileNotFoundError(
            f"{len(faltando)} arquivo(s) referenciados no kfold_config.json não existem mais. "
            "dataset_raw ou dataset_split foi alterado depois do 02_preprocessing.py rodar. "
            "Rode 02_preprocessing.py de novo para regenerar tudo de forma consistente, "
            "sem editar dataset_raw/dataset_split manualmente entre uma rodada e outra."
        )

    print(f"Validação: {total} caminhos do kfold_config.json conferidos, todos existem.")


def criar_modelo(timm_name, tentativas=3):
    for tentativa in range(tentativas):
        try:
            return timm.create_model(timm_name, pretrained=True, num_classes=NUM_CLASSES)
        except Exception as e:
            print(f"  Falha ao baixar pesos pré-treinados ({tentativa + 1}/{tentativas}): {e}")
            if tentativa == tentativas - 1:
                raise
            time.sleep(5)


def avaliar(model, paths, labels, transform, batch_size):
    ds = MilhoDataset(paths, labels, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(DEVICE)

    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            f1_metric.update(model(imgs).argmax(dim=1), lbls)

    return f1_metric.compute().item()


def treinar_fold(modelo, fold_idx, fold_data, cfg, kfold):
    img_size = cfg['img_size']
    batch_size = cfg['batch_size']
    transform = build_transform(img_size)

    train_ds = MilhoDataset(fold_data['train_paths'], fold_data['train_labels'], transform)
    val_ds = MilhoDataset(fold_data['val_paths'], fold_data['val_labels'], transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = criar_modelo(cfg['timm_name']).to(DEVICE)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    label_counts = pd.Series(fold_data['train_labels']).value_counts()
    label_counts = label_counts.reindex(range(NUM_CLASSES), fill_value=1)
    weights = 1.0 / torch.tensor(label_counts.values, dtype=torch.float)
    weights = (weights / weights.sum() * NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(DEVICE)
    acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average='micro').to(DEVICE)

    for name, param in model.named_parameters():
        if not any(k in name for k in ['classifier', 'head', 'fc']):
            param.requires_grad = False
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LR_HEAD, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    epoch_inicial = 0
    best_f1 = 0.0
    patience_counter = 0
    best_weights = None
    history = []
    backbone_unfrozen = False

    ckpt_path = checkpoint_path(modelo, fold_idx)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        backbone_unfrozen = ckpt['backbone_unfrozen']
        if backbone_unfrozen:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        if 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        epoch_inicial = ckpt['epoch'] + 1
        best_f1 = ckpt['best_f1']
        patience_counter = ckpt['patience_counter']
        best_weights = ckpt['best_weights']
        history = ckpt['history']
        print(f"  Retomando fold {fold_idx + 1} da época {epoch_inicial + 1} (checkpoint encontrado)")

    print(f"\n  Fold {fold_idx + 1}/{N_FOLDS} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    for epoch in range(epoch_inicial, NUM_EPOCHS):
        if epoch == WARMUP_EPOCHS and not backbone_unfrozen:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
            backbone_unfrozen = True
            print(f"  Backbone descongelado — lr={LR_FULL}")

        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        f1_metric.reset()
        acc_metric.reset()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                    outputs = model(imgs)
                    val_loss += criterion(outputs, labels).item()
                f1_metric.update(outputs.argmax(dim=1), labels)
                acc_metric.update(outputs.argmax(dim=1), labels)

        val_f1 = f1_metric.compute().item()
        val_acc = acc_metric.compute().item()
        avg_tl = train_loss / len(train_loader)
        avg_vl = val_loss / len(val_loader)
        history.append({'epoch': epoch + 1, 'train_loss': avg_tl, 'val_loss': avg_vl,
                         'val_f1': val_f1, 'val_acc': val_acc})

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        print(f"  Epoch {epoch + 1:>3} | loss: {avg_tl:.4f} | val_loss: {avg_vl:.4f} | "
              f"val_f1: {val_f1:.4f} | patience: {patience_counter}/{PATIENCE}")

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'backbone_unfrozen': backbone_unfrozen,
            'best_f1': best_f1,
            'patience_counter': patience_counter,
            'best_weights': best_weights,
            'history': history,
        }, ckpt_path)

        if patience_counter >= PATIENCE:
            print(f"  Early stopping na época {epoch + 1}. Melhor F1: {best_f1:.4f}")
            break

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), final_model_path(modelo, fold_idx))
    ckpt_path.unlink(missing_ok=True)

    test_f1 = avaliar(model, kfold['test_paths'], kfold['test_labels'], transform, batch_size)
    print(f"  Fold {fold_idx + 1} concluído — val F1 {best_f1:.4f} | test F1 {test_f1:.4f}")

    with open(fold_resultado_path(modelo, fold_idx), 'w', encoding='utf-8') as f:
        json.dump({'fold': fold_idx, 'val_f1': best_f1, 'test_f1': test_f1, 'history': history},
                   f, indent=2, ensure_ascii=False)

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return best_f1, test_f1


def treinar_modelo(modelo):
    if modelo not in MODELS:
        raise ValueError(f"Modelo '{modelo}' não existe em MODELS: {list(MODELS.keys())}")
    cfg = MODELS[modelo]

    print(f"Device: {DEVICE}")
    print(f"Modelo: {modelo} ({cfg['timm_name']}) | batch={cfg['batch_size']} | img={cfg['img_size']}")

    if resultado_path(modelo).exists():
        print(f"\n{resultado_path(modelo)} já existe — modelo já treinado e avaliado.")
        print("Apague o arquivo se quiser retreinar do zero.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)

    inicio = time.time()
    val_f1s, test_f1s = [], []

    for fold_idx, fold_data in enumerate(kfold['folds']):
        fr_path = fold_resultado_path(modelo, fold_idx)
        if fr_path.exists():
            with open(fr_path) as f:
                fr = json.load(f)
            print(f"\n  Fold {fold_idx + 1} já concluído — val F1 {fr['val_f1']:.4f} | test F1 {fr['test_f1']:.4f} (pulando)")
            val_f1s.append(fr['val_f1'])
            test_f1s.append(fr['test_f1'])
            continue

        val_f1, test_f1 = treinar_fold(modelo, fold_idx, fold_data, cfg, kfold)
        val_f1s.append(val_f1)
        test_f1s.append(test_f1)

    tempo_treino_min = (time.time() - inicio) / 60

    resultado = {
        'modelo': modelo,
        'timm_name': cfg['timm_name'],
        'val_f1_macro_media': float(np.mean(val_f1s)),
        'val_f1_macro_std': float(np.std(val_f1s)),
        'val_f1_por_fold': val_f1s,
        'test_f1_macro_media': float(np.mean(test_f1s)),
        'test_f1_macro_std': float(np.std(test_f1s)),
        'test_f1_por_fold': test_f1s,
        'tempo_treino_min': round(tempo_treino_min, 1),
    }

    with open(resultado_path(modelo), 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(f"\n{modelo}: val F1 {resultado['val_f1_macro_media']:.4f}±{resultado['val_f1_macro_std']:.4f} | "
          f"test F1 {resultado['test_f1_macro_media']:.4f}±{resultado['test_f1_macro_std']:.4f} | "
          f"{tempo_treino_min:.0f} min")
    print(f"Resultado salvo em: {resultado_path(modelo)}")
    return resultado
