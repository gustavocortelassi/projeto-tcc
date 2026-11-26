"""
Controle do atalho de cor (apontamento 6 da banca). Não retreina nada e não
sobrescreve resultados anteriores.

Pergunta: os modelos finais dependem da cor/fonte da imagem ou da morfologia?

1. Robustez: avalia o ensemble de cada arquitetura no MESMO teste fixo em três
   condições:
     - original
     - tons de cinza (remove toda a informação de cor)
     - perturbação forte de cor (matiz, saturação e brilho; semente fixa)
   Se o desempenho se mantém sem cor, a decisão não depende do atalho de cor.

2. Acerto por fonte dentro da mesma classe: nas classes com imagens de mais de
   uma fonte, compara a acurácia por fonte (usa predicoes_teste_<modelo>.npz,
   gerado pelo 06b; sem inferência). Acerto alto em todas as fontes da mesma
   classe indica que o modelo não está apenas reconhecendo a fonte.

Saídas novas em outputs/results/:
  robustez_cor.json
  robustez_cor_log.txt

Uso:
    python -u 15_teste_robustez_cor.py
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import f1_score, accuracy_score
import timm

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from treino_core import MilhoDataset, build_transform

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEMENTE = 42
FONTES = ('usuario', 'professor', 'externo')
GRUPO_V1_V4 = ['V1', 'V2', 'V3', 'V4']


class Tee:
    def __init__(self, caminho):
        self.arq = open(caminho, 'w', encoding='utf-8')
        self.tela = sys.stdout

    def write(self, s):
        self.tela.write(s); self.tela.flush()
        self.arq.write(s); self.arq.flush()

    def flush(self):
        self.tela.flush(); self.arq.flush()


RESULT_DIR.mkdir(parents=True, exist_ok=True)
sys.stdout = Tee(RESULT_DIR / 'robustez_cor_log.txt')
T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}")


def condicoes(img_size):
    base = build_transform(img_size)
    return {
        'original': base,
        'tons_de_cinza': T.Compose([T.Grayscale(num_output_channels=3), base]),
        'cor_perturbada': T.Compose([T.ColorJitter(brightness=0.4, saturation=0.6, hue=0.08), base]),
    }


def carregar_folds(modelo, cfg):
    ms = []
    for fold in range(1, N_FOLDS + 1):
        p = MODEL_DIR / f"{modelo}_fold{fold}.pt"
        if not p.exists():
            log(f"  [AVISO] {p.name} não encontrado")
            continue
        m = timm.create_model(cfg['timm_name'], pretrained=False, num_classes=NUM_CLASSES)
        m.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
        ms.append(m.to(DEVICE).eval())
    return ms


def prever(ms, loader):
    out = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            p = sum(Fn.softmax(m(imgs), dim=1) for m in ms) / len(ms)
            out.append(p.float().cpu())
    return torch.cat(out).numpy().argmax(axis=1)


def fonte_de(caminho):
    nome = Path(caminho).name
    for f in FONTES:
        if nome.startswith(f + '_'):
            return f
    return 'outra'


def main():
    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)
    paths, y = kfold['test_paths'], np.array(kfold['test_labels'])
    idx_v14 = [LABEL_TO_IDX[c] for c in GRUPO_V1_V4]
    saida = {'robustez': {}, 'por_fonte': {}}

    # ── 1. Robustez à remoção/perturbação de cor ─────────────────────────────
    log(f"Dispositivo: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
    for modelo, cfg in MODELS.items():
        ms = carregar_folds(modelo, cfg)
        if not ms:
            continue
        saida['robustez'][modelo] = {}
        for nome, tf in condicoes(cfg['img_size']).items():
            torch.manual_seed(SEMENTE)
            ds = MilhoDataset(paths, y.tolist(), tf)
            loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
            pred = prever(ms, loader)
            r = {'f1_macro': float(f1_score(y, pred, average='macro')),
                 'acuracia': float(accuracy_score(y, pred)),
                 'f1_macro_v1_v4': float(f1_score(y, pred, labels=idx_v14, average='macro')),
                 'erros': int((pred != y).sum())}
            saida['robustez'][modelo][nome] = r
            log(f"  {modelo:<15} {nome:<15} F1={r['f1_macro']:.4f}  acc={r['acuracia']:.4f}  "
                f"F1 V1-V4={r['f1_macro_v1_v4']:.4f}  erros={r['erros']}")
        del ms
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # ── 2. Acerto por fonte dentro da mesma classe ───────────────────────────
    print(f"\n{'=' * 70}\nACERTO POR FONTE NAS CLASSES COM MAIS DE UMA FONTE (ensemble, teste)\n{'=' * 70}")
    for modelo in MODELS:
        npz = RESULT_DIR / f"predicoes_teste_{modelo}.npz"
        if not npz.exists():
            print(f"  {modelo}: {npz.name} não encontrado (rode o 06b antes)")
            continue
        d = np.load(npz, allow_pickle=False)
        pred, lab, pts = d['preds_ensemble'], d['labels'], d['paths']
        cont = defaultdict(lambda: [0, 0])
        for p, l, c in zip(pred, lab, pts):
            k = (IDX_TO_LABEL[int(l)], fonte_de(str(c)))
            cont[k][0] += int(p == l); cont[k][1] += 1
        por_classe = defaultdict(dict)
        for (cls, fonte), (ac, n) in cont.items():
            por_classe[cls][fonte] = {'acertos': ac, 'n': n}
        multi = {c: v for c, v in por_classe.items() if len(v) > 1}
        saida['por_fonte'][modelo] = multi
        print(f"\n  {modelo}")
        for cls in sorted(multi, key=lambda c: list(LABEL_TO_IDX).index(c)):
            partes = '  '.join(f"{f}: {v['acertos']}/{v['n']}" for f, v in sorted(multi[cls].items()))
            print(f"    {cls:<6} {partes}")

    with open(RESULT_DIR / 'robustez_cor.json', 'w', encoding='utf-8') as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    log(f"Salvo em: {RESULT_DIR / 'robustez_cor.json'}")


if __name__ == "__main__":
    main()
