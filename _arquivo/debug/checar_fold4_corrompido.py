"""
Verifica se alguma imagem usada no fold 4 (treino + validação) está
corrompida — suspeita levantada pela queda de energia na rodada original
que gerou o primeiro resultado ruim desse fold.

Uso:
    python checar_fold4_corrompido.py
"""

import json
from pathlib import Path
from PIL import Image

with open('outputs/results/kfold_config.json', encoding='utf-8') as f:
    kfold = json.load(f)

fold4 = kfold['folds'][3]  # índice 3 = fold 4
caminhos = fold4['train_paths'] + fold4['val_paths']

ruins = []
for path in caminhos:
    try:
        img = Image.open(path)
        img.verify()
    except Exception as e:
        ruins.append((path, str(e)))

print(f"{len(ruins)} arquivo(s) com problema, de {len(caminhos)} total")
for p, e in ruins[:20]:
    print(f"  {p}: {e}")

if not ruins:
    print("\nNenhum arquivo corrompido encontrado no fold 4.")
