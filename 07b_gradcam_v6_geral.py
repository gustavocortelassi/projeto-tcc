"""
Grad-CAM (ResNet50, EfficientNetV2) e mapa de atenção da última camada
(ViT) nas imagens de V6 classificadas erradas no conjunto de teste —
qualquer classe errada, não só V7 (critério mais amplo que o dataset
anterior, onde V6->V7 era o padrão dominante; nesse dataset novo esse
padrão específico não se repetiu com a mesma força, mas V6 continua
sendo a classe mais difícil em todos os modelos).

Swin fica de fora aqui — já coberto por 08_comparar_folds_atencao.py
(comparação fold estável x instável, achado central deste dataset).
MobileViT fica de fora por ser híbrido CNN+Transformer, sem adaptação
direta pra nenhuma das duas técnicas.

Uso:
    python 07b_gradcam_v6_geral.py
"""

import sys
import json
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
OUT_DIR_VIS = PLOT_DIR / 'interpretabilidade'

MODELOS_CNN = ['resnet50', 'efficientnetv2']
MODELOS_TRANSFORMER = ['vit']


def carregar_fold1(modelo, cfg):
    path = MODEL_DIR / f"{modelo}_fold1.pt"
    if not path.exists():
        return None
    m = timm.create_model(cfg['timm_name'], pretrained=False, num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.to(DEVICE).eval()
    return m


def encontrar_v6_errado(modelo_fold1, kfold, transform):
    """Retorna [(path, classe_predita_idx), ...] para V6 que o PRÓPRIO fold1
    errou — mesmo modelo usado no Grad-CAM/atenção, não o ensemble (senão o
    título pode não bater com o que esse modelo específico previu)."""
    v6_idx = LABEL_TO_IDX['V6']

    encontrados = []
    for path, label in zip(kfold['test_paths'], kfold['test_labels']):
        if label != v6_idx:
            continue
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = modelo_fold1(img_tensor).argmax(dim=1).item()
        if pred != v6_idx:
            encontrados.append((path, pred))

    return encontrados


def desnormalizar(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def salvar_comparacao(img_tensor, mapa, titulo, destino):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    img_np = desnormalizar(img_tensor.squeeze(0))

    axes[0].imshow(img_np)
    axes[0].set_title('Original (V6)')
    axes[0].axis('off')

    axes[1].imshow(img_np)
    axes[1].imshow(mapa, cmap='jet', alpha=0.5)
    axes[1].set_title(titulo)
    axes[1].axis('off')

    plt.tight_layout()

    for tentativa in range(3):
        try:
            plt.savefig(destino, dpi=120)
            break
        except OSError as e:
            print(f"  [AVISO] Falha ao salvar {destino.name} (tentativa {tentativa + 1}/3): {e}")
            if tentativa == 2:
                print(f"  [ERRO] Desistindo de {destino.name} após 3 tentativas.")
            else:
                import time
                time.sleep(1)
    plt.close(fig)


# ─── Grad-CAM (CNNs) ────────────────────────────────────────────────────────

def gradcam(model, img_tensor, target_class):
    model.zero_grad()
    features = model.forward_features(img_tensor)
    features.retain_grad()
    logits = model.forward_head(features)
    logits[0, target_class].backward()

    grads = features.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = Fn.relu((weights * features).sum(dim=1, keepdim=True))
    cam = Fn.interpolate(cam, size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def rodar_gradcam(m, modelo, cfg, casos, transform):
    destino_dir = OUT_DIR_VIS / modelo
    destino_dir.mkdir(parents=True, exist_ok=True)

    for i, (path, pred_idx) in enumerate(casos):
        classe_predita = IDX_TO_LABEL[pred_idx]
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        cam = gradcam(m, img_tensor, pred_idx)
        salvar_comparacao(img_tensor, cam, f'{modelo} — Grad-CAM (classe {classe_predita})',
                           destino_dir / f'v6_errado_{i:02d}_{classe_predita}.png')

    print(f"  {modelo}: {len(casos)} imagens salvas em {destino_dir}")


# ─── Attention Rollout completo (ViT) ───────────────────────────────────────
# Diferente do Swin (hierárquico, token count muda por estágio — rollout
# acumulado não é válido lá, por isso o script de fold usa só última camada),
# o ViT mantém o mesmo número de tokens em toda camada. Rollout completo
# (Abnar & Zuidema 2020) é válido aqui e dá um mapa bem mais rico que uma
# camada só — resolve o problema de "só uma bolinha" que atenção de última
# camada isolada tende a mostrar (fenômeno de "attention sink" documentado
# em ViT: poucos tokens concentram atenção desproporcional independente do
# conteúdo visual).

def forcar_atencao_eager(model):
    for module in model.modules():
        if hasattr(module, 'fused_attn'):
            module.fused_attn = False


def attention_rollout_completo(model, img_tensor):
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
        raise RuntimeError(
            "Nenhuma matriz de atenção capturada — build do timm pode usar "
            "scaled_dot_product_attention fundido sem expor attn_drop."
        )

    resultado = None
    for attn in capturadas:
        attn = attn.mean(dim=1)  # média das cabeças: (B, N, N)
        attn = attn + torch.eye(attn.size(-1), device=attn.device)  # conexão residual
        attn = attn / attn.sum(dim=-1, keepdim=True)
        resultado = attn if resultado is None else attn @ resultado

    return resultado[0]  # (N, N)


def mapa_de_v_para_grade(attn_final, grid_size, img_size):
    mapa = attn_final.mean(dim=0)[-grid_size * grid_size:]
    mapa = mapa.reshape(grid_size, grid_size).cpu().numpy()
    mapa = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8)
    mapa_img = Image.fromarray((mapa * 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)
    return np.array(mapa_img) / 255.0


def rodar_atencao(m, modelo, cfg, casos, transform):
    destino_dir = OUT_DIR_VIS / modelo
    destino_dir.mkdir(parents=True, exist_ok=True)

    grid_size = cfg['img_size'] // 16

    for i, (path, pred_idx) in enumerate(casos):
        classe_predita = IDX_TO_LABEL[pred_idx]
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        try:
            attn_final = attention_rollout_completo(m, img_tensor)
            mapa = mapa_de_v_para_grade(attn_final, grid_size, cfg['img_size'])
        except Exception as e:
            print(f"  [ERRO] {modelo} imagem {i}: {e}")
            continue
        salvar_comparacao(img_tensor, mapa, f'{modelo} — Attention Rollout (previu {classe_predita})',
                           destino_dir / f'v6_errado_{i:02d}_{classe_predita}.png')

    print(f"  {modelo}: processado, ver {destino_dir}")


def main():
    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)

    OUT_DIR_VIS.mkdir(parents=True, exist_ok=True)

    for modelo in MODELOS_CNN + MODELOS_TRANSFORMER:
        cfg = MODELS[modelo]
        print(f"\n{'=' * 60}\n{modelo}\n{'=' * 60}")

        m = carregar_fold1(modelo, cfg)
        if m is None:
            print("  fold1 não encontrado, pulando")
            continue

        transform = build_transform(cfg['img_size'])
        casos = encontrar_v6_errado(m, kfold, transform)
        print(f"  {len(casos)} imagens de V6 que o fold1 classificou errado "
              f"(classes previstas: {sorted(set(IDX_TO_LABEL[p] for _, p in casos))})")

        if not casos:
            del m
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            continue

        if modelo in MODELOS_CNN:
            rodar_gradcam(m, modelo, cfg, casos, transform)
        else:
            rodar_atencao(m, modelo, cfg, casos, transform)

        del m
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nImagens salvas em: {OUT_DIR_VIS}")


if __name__ == "__main__":
    main()
