"""
Grad-CAM em exemplos classificados CORRETAMENTE, com alta confiança —
contraste com os exemplos de erro já gerados (07b_gradcam_v6_geral.py).
Útil para a figura do artigo: mostrar como a atenção se comporta quando
o modelo acerta com confiança, não só quando erra.

Uso:
    python 14_gradcam_acertos.py --modelo resnet50
    python 14_gradcam_acertos.py --modelo efficientnetv2 --classe V1
    python 14_gradcam_acertos.py --modelo resnet50 --top 8
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
import matplotlib.pyplot as plt
import timm

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from treino_core import build_transform

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR_VIS = PLOT_DIR / 'interpretabilidade' / 'acertos'


def gradcam(model, img_tensor, target_class):
    model.zero_grad()
    features = model.forward_features(img_tensor)
    features.retain_grad()
    logits = model.forward_head(features)
    probs = Fn.softmax(logits, dim=1)
    confianca = probs[0, target_class].item()
    logits[0, target_class].backward()

    grads = features.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = Fn.relu((weights * features).sum(dim=1, keepdim=True))
    cam = Fn.interpolate(cam, size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, confianca


def desnormalizar(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def salvar_comparacao(img_tensor, mapa, titulo, destino):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    img_np = desnormalizar(img_tensor.squeeze(0))

    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(img_np)
    axes[1].imshow(mapa, cmap='jet', alpha=0.5)
    axes[1].set_title(titulo)
    axes[1].axis('off')

    plt.tight_layout()
    for tentativa in range(3):
        try:
            plt.savefig(destino, dpi=150)
            break
        except OSError as e:
            print(f"  [AVISO] Falha ao salvar {destino.name} (tentativa {tentativa + 1}/3): {e}")
            if tentativa < 2:
                import time
                time.sleep(1)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelo', required=True, choices=['resnet50', 'efficientnetv2'],
                         help='Só CNNs — Grad-CAM não se aplica a ViT/Swin/MobileViT.')
    parser.add_argument('--classe', default=None,
                         help='Filtra por uma classe específica (ex.: V1, R5). Sem isso, pega a melhor de qualquer classe.')
    parser.add_argument('--top', type=int, default=6, help='Quantos exemplos gerar (padrão: 6).')
    args = parser.parse_args()

    cfg = MODELS[args.modelo]
    transform = build_transform(cfg['img_size'])

    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)

    print(f"Carregando {args.modelo} (fold1)...")
    m = timm.create_model(cfg['timm_name'], pretrained=False, num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(MODEL_DIR / f"{args.modelo}_fold1.pt", map_location=DEVICE, weights_only=True))
    m.to(DEVICE).eval()

    print("Rodando inferência no conjunto de teste, procurando acertos de alta confiança...")
    candidatos = []
    for path, label in zip(kfold['test_paths'], kfold['test_labels']):
        classe_nome = IDX_TO_LABEL[label]
        if args.classe and classe_nome != args.classe:
            continue

        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = m(img_tensor)
            probs = Fn.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confianca = probs[0, pred].item()

        if pred == label:
            candidatos.append((path, label, confianca))

    candidatos.sort(key=lambda x: -x[2])  # maior confiança primeiro
    print(f"{len(candidatos)} acertos encontrados. Gerando Grad-CAM dos {args.top} de maior confiança...\n")

    OUT_DIR_VIS.mkdir(parents=True, exist_ok=True)
    for i, (path, label, _) in enumerate(candidatos[:args.top]):
        classe_nome = IDX_TO_LABEL[label]
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        mapa, confianca = gradcam(m, img_tensor, label)
        titulo = f'{args.modelo} — Grad-CAM ({classe_nome})'
        destino = OUT_DIR_VIS / f"{args.modelo}_{classe_nome}_{i:02d}_conf{confianca:.3f}.png"
        salvar_comparacao(img_tensor, mapa, titulo, destino)
        print(f"  {destino.name}")

    print(f"\nSalvo em: {OUT_DIR_VIS}")


if __name__ == "__main__":
    main()
