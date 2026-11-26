"""
Compara o mapa de atenção do Swin entre dois folds — por padrão, o mais
estável (maior F1) e o mais instável (menor F1) — nas imagens de teste
onde o fold bom acerta e o fold ruim erra.

Motivação: o achado central da réplica não foi mais "erra V6 como V7" (não
se repetiu com força nesse dataset), foi a VARIÂNCIA entre folds do Swin
(desvio ~0,054, ~8x maior que o ViT). Esse script transforma esse número
em evidência visual: se a atenção do fold ruim for visivelmente mais
dispersa/inconsistente que a do fold bom, exatamente nas imagens em que
ele erra e o outro acerta, é evidência direta de que a instabilidade tem
assinatura visual, não é só ruído de métrica.

Usa a atenção da última camada (o Swin é hierárquico e com janela, então
rollout acumulado não se aplica).

Uso:
    python 08_comparar_folds_atencao.py --fold-bom 1 --fold-ruim 3
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
OUT_DIR_VIS = PLOT_DIR / 'interpretabilidade' / 'comparacao_folds_swin'
MODELO = 'swin'
MAX_IMAGENS = 12  # limite de exemplos gerados, para não produzir centenas


def carregar_modelo_fold(modelo, cfg, fold_idx):
    m = timm.create_model(cfg['timm_name'], pretrained=False, num_classes=NUM_CLASSES)
    path = MODEL_DIR / f"{modelo}_fold{fold_idx}.pt"
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.to(DEVICE).eval()
    return m


def prever_todas(modelo, kfold, transform, batch_size):
    preds = []
    for path in kfold['test_paths']:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = modelo(img_tensor)
        preds.append(logits.argmax(dim=1).item())
    return preds


def forcar_atencao_eager(model):
    for module in model.modules():
        if hasattr(module, 'fused_attn'):
            module.fused_attn = False


def atencao_ultima_camada(model, img_tensor):
    capturadas = []

    def hook_attn_drop(module, input, output):
        capturadas.append(input[0].detach())

    handles = []
    for name, module in model.named_modules():
        if name.endswith('attn_drop'):
            handles.append(module.register_forward_hook(hook_attn_drop))

    forcar_atencao_eager(model)
    with torch.no_grad():
        model(img_tensor)
    for h in handles:
        h.remove()

    if not capturadas:
        raise RuntimeError("Nenhuma atenção capturada — ver 07_interpretabilidade.py para diagnóstico.")

    ultima_attn = capturadas[-1].mean(dim=1)  # média das cabeças
    return ultima_attn[0]


def mapa_de_v_para_grade(attn_final, grid_size, img_size):
    mapa = attn_final.mean(dim=0)[-grid_size * grid_size:]
    mapa = mapa.reshape(grid_size, grid_size).cpu().numpy()
    mapa = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8)
    mapa_img = Image.fromarray((mapa * 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)
    return np.array(mapa_img) / 255.0


def desnormalizar(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def salvar_comparacao_tripla(img_tensor, mapa_bom, mapa_ruim, fold_bom, fold_ruim, destino):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    img_np = desnormalizar(img_tensor.squeeze(0))

    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(img_np)
    axes[1].imshow(mapa_bom, cmap='jet', alpha=0.5)
    axes[1].set_title(f'Fold {fold_bom} (estável) — acerta')
    axes[1].axis('off')

    axes[2].imshow(img_np)
    axes[2].imshow(mapa_ruim, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Fold {fold_ruim} (instável) — erra')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(destino, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold-bom', type=int, required=True)
    parser.add_argument('--fold-ruim', type=int, required=True)
    args = parser.parse_args()

    cfg = MODELS[MODELO]
    transform = build_transform(cfg['img_size'])
    grid_size = cfg['img_size'] // 32  # Swin: patch4 + 3 estágios de downsample adicional

    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)

    print(f"Carregando fold {args.fold_bom} (estável) e fold {args.fold_ruim} (instável)...")
    modelo_bom = carregar_modelo_fold(MODELO, cfg, args.fold_bom)
    modelo_ruim = carregar_modelo_fold(MODELO, cfg, args.fold_ruim)

    print("Rodando inferência dos dois folds no conjunto de teste...")
    preds_bom = prever_todas(modelo_bom, kfold, transform, cfg['batch_size'])
    preds_ruim = prever_todas(modelo_ruim, kfold, transform, cfg['batch_size'])

    labels = kfold['test_labels']
    candidatos = [
        (path, label) for path, label, p_bom, p_ruim
        in zip(kfold['test_paths'], labels, preds_bom, preds_ruim)
        if p_bom == label and p_ruim != label
    ]

    print(f"\n{len(candidatos)} imagens onde fold {args.fold_bom} acerta e fold {args.fold_ruim} erra "
          f"(de {len(labels)} no teste)")

    if not candidatos:
        print("Nenhuma imagem nesse critério — os dois folds erram/acertam sempre juntos "
              "nesse conjunto de teste. Considerar comparar em cima do conjunto de validação "
              "de cada fold, não do teste fixo.")
        return

    OUT_DIR_VIS.mkdir(parents=True, exist_ok=True)
    for i, (path, label) in enumerate(candidatos[:MAX_IMAGENS]):
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        attn_bom = atencao_ultima_camada(modelo_bom, img_tensor)
        mapa_bom = mapa_de_v_para_grade(attn_bom, grid_size, cfg['img_size'])

        attn_ruim = atencao_ultima_camada(modelo_ruim, img_tensor)
        mapa_ruim = mapa_de_v_para_grade(attn_ruim, grid_size, cfg['img_size'])

        classe = IDX_TO_LABEL[label]
        destino = OUT_DIR_VIS / f"{classe}_{i:02d}.png"
        salvar_comparacao_tripla(img_tensor, mapa_bom, mapa_ruim, args.fold_bom, args.fold_ruim, destino)

    n_salvas = min(len(candidatos), MAX_IMAGENS)
    print(f"\n{n_salvas} comparação(ões) salva(s) em: {OUT_DIR_VIS}")
    if len(candidatos) > MAX_IMAGENS:
        print(f"({len(candidatos) - MAX_IMAGENS} imagens adicionais no critério não foram geradas, "
              f"limite de {MAX_IMAGENS})")


if __name__ == "__main__":
    main()
