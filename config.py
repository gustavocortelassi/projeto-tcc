# =============================================================================
# config.py – Projeto Final (Fenologia do Milho)
# Dataset unificado: fotos próprias + banco do professor + banco externo
# (V1/V3/V4, filtrado por espécie e deduplicado por hash de conteúdo)
# =============================================================================

from pathlib import Path

# ─── Caminhos ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\Gustavo\Desktop\Projeto Final")
RAW_DIR      = PROJECT_ROOT / "dataset_unificado"
SPLIT_DIR    = PROJECT_ROOT / "dataset_split"
OUT_DIR      = PROJECT_ROOT / "outputs"
MODEL_DIR    = OUT_DIR / "models"
PLOT_DIR     = OUT_DIR / "plots"
RESULT_DIR   = OUT_DIR / "results"

# ─── Classes (Escala Ritchie) ────────────────────────────────────────────────
# União das classes das 3 fontes já mescladas em dataset_unificado. Sem V8 —
# nunca apareceu em nenhuma das 3 fontes deste projeto.
CLASSES = ['R1', 'R2', 'R3_R4', 'R5', 'R6',
           'V1', 'V2', 'V3', 'V4', 'V6', 'V7', 'V12', 'VT']
EXCLUDED_CLASSES = []
CLASSES_TREINO   = [c for c in CLASSES if c not in EXCLUDED_CLASSES]
NUM_CLASSES      = len(CLASSES_TREINO)

# Arquivos especificamente excluídos por instabilidade confirmada em disco.
ARQUIVOS_EXCLUIDOS = set()

# ─── Limiares de adequação por classe ────────────────────────────────────────
LIMIAR_ADEQUADO = 500
LIMIAR_LIMITADO = 100

# ─── Balanceamento ────────────────────────────────────────────────────────────
# Alvo simétrico (não só piso) para não confundir volume de treino com
# estádio vegetativo x reprodutivo (H1), valor derivado do multiplicador de
# augmentation máximo aceito para a classe mais escassa.
TARGET_N_AUG = 500
TETO_MAX = TARGET_N_AUG

# ─── Hiperparâmetros globais ──────────────────────────────────────────────────
SEED        = 42
IMG_SIZE    = 384
BATCH_SIZE  = 32
NUM_EPOCHS  = 50
PATIENCE    = 10
NUM_WORKERS = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Modelos do benchmark ─────────────────────────────────────────────────────
MODELS = {
    "resnet50"       : {"timm_name": "resnet50",                      "img_size": 224, "batch_size": 32},
    "efficientnetv2" : {"timm_name": "tf_efficientnetv2_s",           "img_size": 384, "batch_size": 12},
    "vit"            : {"timm_name": "vit_base_patch16_224",          "img_size": 224, "batch_size": 24},
    "swin"           : {"timm_name": "swin_base_patch4_window12_384", "img_size": 384, "batch_size": 8},
    "mobilevit"      : {"timm_name": "mobilevitv2_150",               "img_size": 256, "batch_size": 24},
}

# ─── Warmup / fine-tuning ──────────────────────────────────────────────────────
LR_HEAD       = 1e-3
LR_FULL       = 5e-5
WARMUP_EPOCHS = 3

# ─── Split / K-Fold ────────────────────────────────────────────────────────────
LABEL_TO_IDX  = {cls: i for i, cls in enumerate(CLASSES_TREINO)}
IDX_TO_LABEL  = {i: cls for cls, i in LABEL_TO_IDX.items()}
AUGMENTED_DIR = SPLIT_DIR / "augmented"
KFOLD_CONFIG  = RESULT_DIR / "kfold_config.json"
N_FOLDS       = 5
TEST_RATIO    = 0.15

# ─── Checagem rápida do ambiente ──────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    print(f"PROJECT_ROOT   : {PROJECT_ROOT}")
    print(f"RAW_DIR        : {RAW_DIR}  (existe: {RAW_DIR.exists()})")
    print(f"Classes treino : {CLASSES_TREINO} ({NUM_CLASSES})")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
