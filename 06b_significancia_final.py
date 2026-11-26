"""
Análise estatística final (H1, H2, H3) sobre os modelos já treinados.
Não retreina nada e não sobrescreve resultados anteriores do pipeline.

1. Inferência de cada modelo de fold sobre o teste fixo. Salva um .npz por
   modelo assim que ele termina — se o script for interrompido, ao rodar de
   novo os modelos já concluídos são carregados do .npz e pulados.
   Confere que o ensemble (média das softmax) reproduz o 05_avaliar_final.py.
2. H1 — Friedman (+ W de Kendall) sobre F1 por fold: macro geral e V1–V4.
       McNemar exato entre pares (ensemble, imagem a imagem), geral e V1–V4,
       com correção de Holm.
3. H2 — parâmetros, tamanho em disco e latência de inferência (batch 1).
4. H3 — desvio padrão do F1 por classe entre os folds, por família.

Mostra o progresso em tempo real e grava tudo também em
outputs/results/significancia_log.txt (não precisa redirecionar com >).

Saídas novas em outputs/results/:
  predicoes_teste_<modelo>.npz
  significancia_final.json
  significancia_log.txt

Uso:
    python -u 06b_significancia_final.py
    python -u 06b_significancia_final.py --refazer   (ignora .npz já salvos)
"""

import sys
import json
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.metrics import f1_score
import timm

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from treino_core import MilhoDataset, build_transform

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FAMILIA = {'resnet50': 'CNN', 'efficientnetv2': 'CNN',
           'vit': 'Atenção', 'swin': 'Atenção', 'mobilevit': 'Híbrido'}
GRUPO_V1_V4 = ['V1', 'V2', 'V3', 'V4']
CLASSES_H3 = ['V4', 'V6', 'V2', 'R3_R4']   # menor nº de imagens originais
N_WARMUP, N_MEDICAO = 10, 50
WORKERS_INFERENCIA = 0   # 0 = carrega no próprio processo (evita respawn lento no Windows)
REFAZER = '--refazer' in sys.argv


# ─── Log na tela e em arquivo, linha a linha ────────────────────────────────
class Tee:
    def __init__(self, caminho):
        self.arq = open(caminho, 'w', encoding='utf-8')
        self.tela = sys.stdout

    def write(self, s):
        self.tela.write(s)
        self.tela.flush()
        self.arq.write(s)
        self.arq.flush()

    def flush(self):
        self.tela.flush()
        self.arq.flush()


RESULT_DIR.mkdir(parents=True, exist_ok=True)
sys.stdout = Tee(RESULT_DIR / 'significancia_log.txt')
T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}")


# ─── Estatística ─────────────────────────────────────────────────────────────
def holm(pvalores):
    """Correção de Holm: retorna p ajustados na mesma ordem."""
    m = len(pvalores)
    ordem = np.argsort(pvalores)
    ajust = np.empty(m)
    acumulado = 0.0
    for rank, i in enumerate(ordem):
        acumulado = max(acumulado, min(1.0, (m - rank) * pvalores[i]))
        ajust[i] = acumulado
    return ajust.tolist()


def mcnemar_exato(acertos_a, acertos_b):
    b = int(np.sum(acertos_a & ~acertos_b))   # A acerta, B erra
    c = int(np.sum(~acertos_a & acertos_b))   # A erra, B acerta
    p = 1.0 if b + c == 0 else stats.binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, float(p)


def friedman(matriz):
    """matriz: modelos x folds. Retorna chi2, p, W de Kendall."""
    k, n = matriz.shape
    try:
        chi2, p = stats.friedmanchisquare(*matriz)
    except ValueError:
        return None, None, None
    return float(chi2), float(p), float(chi2 / (n * (k - 1)))


# ─── Inferência ──────────────────────────────────────────────────────────────
def carregar(modelo, cfg, fold):
    path = MODEL_DIR / f"{modelo}_fold{fold}.pt"
    if not path.exists():
        return None, None
    m = timm.create_model(cfg['timm_name'], pretrained=False, num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE).eval(), path


def prever(m, loader):
    probs = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
        for imgs, _ in loader:
            probs.append(Fn.softmax(m(imgs.to(DEVICE)), dim=1).float().cpu())
    return torch.cat(probs).numpy()


def medir_latencia(m, img_size):
    x = torch.randn(1, 3, img_size, img_size, device=DEVICE)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
        for _ in range(N_WARMUP):
            m(x)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_MEDICAO):
            m(x)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_MEDICAO * 1000


def inferir_modelo(modelo, cfg, test_paths, test_labels, nomes):
    """Roda os 5 folds de um modelo, salva o .npz e devolve preds + eficiência."""
    ds = MilhoDataset(test_paths, test_labels.tolist(), build_transform(cfg['img_size']))
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False,
                        num_workers=WORKERS_INFERENCIA)
    probs_folds, efic = [], None
    for fold in range(1, N_FOLDS + 1):
        t = time.perf_counter()
        m, path = carregar(modelo, cfg, fold)
        if m is None:
            log(f"  [AVISO] {modelo} fold {fold} não encontrado")
            continue
        probs_folds.append(prever(m, loader))
        if fold == 1:
            efic = {
                'familia': FAMILIA.get(modelo, '?'),
                'parametros_milhoes': sum(p.numel() for p in m.parameters()) / 1e6,
                'tamanho_mb': path.stat().st_size / 2**20,
                'img_size': cfg['img_size'],
                'latencia_ms_batch1': medir_latencia(m, cfg['img_size']),
            }
        log(f"  {modelo} fold {fold}/{N_FOLDS} ok ({time.perf_counter() - t:.1f}s)")
        del m
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    if not probs_folds:
        return None
    probs_folds = np.stack(probs_folds)              # folds x imagens x classes
    preds_folds = probs_folds.argmax(axis=2)
    preds_ens = probs_folds.mean(axis=0).argmax(axis=1)
    np.savez_compressed(RESULT_DIR / f"predicoes_teste_{modelo}.npz",
                        probs_folds=probs_folds, preds_folds=preds_folds,
                        preds_ensemble=preds_ens, labels=test_labels,
                        paths=np.array(test_paths), classes=np.array(nomes),
                        eficiencia=json.dumps(efic, ensure_ascii=False))
    return {'preds_folds': preds_folds, 'preds_ens': preds_ens}, efic


def inferir_todos(test_paths, test_labels):
    nomes = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    resultado, eficiencia = {}, {}
    for modelo, cfg in MODELS.items():
        npz = RESULT_DIR / f"predicoes_teste_{modelo}.npz"
        if npz.exists() and not REFAZER:
            d = np.load(npz, allow_pickle=False)
            if 'eficiencia' in d.files and (d['labels'] == test_labels).all():
                resultado[modelo] = {'preds_folds': d['preds_folds'], 'preds_ens': d['preds_ensemble']}
                eficiencia[modelo] = json.loads(str(d['eficiencia']))
                log(f"{modelo}: carregado de {npz.name} (já concluído antes)")
                erros = int((resultado[modelo]['preds_ens'] != test_labels).sum())
                log(f"  ensemble: {erros} erros (deve bater com o 05_avaliar_final)")
                continue

        log(f"{modelo}: inferência nos {N_FOLDS} folds ({cfg['img_size']}px)...")
        r = inferir_modelo(modelo, cfg, test_paths, test_labels, nomes)
        if r is None:
            continue
        resultado[modelo], eficiencia[modelo] = r
        erros = int((resultado[modelo]['preds_ens'] != test_labels).sum())
        log(f"  ensemble: {erros} erros (deve bater com o 05_avaliar_final)")
    return resultado, eficiencia


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    with open(KFOLD_CONFIG, encoding='utf-8') as f:
        kfold = json.load(f)
    test_paths = kfold['test_paths']
    y = np.array(kfold['test_labels'])
    idx_v14 = [LABEL_TO_IDX[c] for c in GRUPO_V1_V4]
    mask_v14 = np.isin(y, idx_v14)

    log(f"Dispositivo: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU (sem GPU!)'}")
    log(f"Teste: {len(y)} imagens | esperado: mobilevit 9, efficientnetv2 9, resnet50 12, vit 14, swin 30 erros")
    pred, efic = inferir_todos(test_paths, y)
    modelos = list(pred.keys())
    saida = {'gpu': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu',
             'n_teste': int(len(y)), 'n_teste_v1_v4': int(mask_v14.sum())}

    # ── H1: Friedman por fold ────────────────────────────────────────────────
    f1_geral = np.array([[f1_score(y, p, average='macro') for p in pred[m]['preds_folds']]
                         for m in modelos])
    f1_v14 = np.array([[f1_score(y, p, labels=idx_v14, average='macro') for p in pred[m]['preds_folds']]
                       for m in modelos])

    print(f"\n{'=' * 70}\nH1 — F1 POR FOLD (modelo individual de cada fold, teste fixo)\n{'=' * 70}")
    for nome_m, chave, mat in [('F1 macro geral', 'geral', f1_geral), ('F1 macro V1–V4', 'v1_v4', f1_v14)]:
        print(f"\n  {nome_m}:")
        for m, linha in zip(modelos, mat):
            print(f"    {m:<16} {' '.join(f'{v:.4f}' for v in linha)}   "
                  f"média={linha.mean():.4f} dp={linha.std(ddof=1):.4f}")
        chi2, p, w = friedman(mat)
        if chi2 is not None:
            print(f"    Friedman: chi2={chi2:.3f}, p={p:.4f}, W de Kendall={w:.3f}")
        saida[f'friedman_{chave}'] = {'f1_por_fold': {m: l.tolist() for m, l in zip(modelos, mat)},
                                      'chi2': chi2, 'p': p, 'kendall_w': w}

    # ── H1: McNemar pareado ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}\nH1 — McNEMAR EXATO ENTRE PARES (ensemble, imagem a imagem, Holm)\n{'=' * 70}")
    for escopo, mask in [('geral', np.ones_like(y, dtype=bool)), ('v1_v4', mask_v14)]:
        linhas = []
        for a, b in combinations(modelos, 2):
            acc_a = pred[a]['preds_ens'][mask] == y[mask]
            acc_b = pred[b]['preds_ens'][mask] == y[mask]
            nb, nc, p = mcnemar_exato(acc_a, acc_b)
            linhas.append({'a': a, 'b': b, 'a_acerta_b_erra': nb, 'a_erra_b_acerta': nc, 'p': p})
        for l, pa in zip(linhas, holm([l['p'] for l in linhas])):
            l['p_holm'] = pa
        print(f"\n  Escopo: {escopo} (n={int(mask.sum())})")
        for l in linhas:
            marca = " *" if l['p_holm'] < 0.05 else ""
            print(f"    {l['a']:<15} vs {l['b']:<15} b={l['a_acerta_b_erra']:>3} c={l['a_erra_b_acerta']:>3} "
                  f"p={l['p']:.4f} p_holm={l['p_holm']:.4f}{marca}")
        saida[f'mcnemar_{escopo}'] = linhas

    # ── H2: eficiência ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}\nH2 — EFICIÊNCIA (fold 1, batch 1, {N_MEDICAO} medições)\n{'=' * 70}")
    for m in modelos:
        e = efic[m]
        e['f1_ensemble'] = float(f1_score(y, pred[m]['preds_ens'], average='macro'))
        print(f"  {m:<16} {e['familia']:<8} {e['parametros_milhoes']:>7.2f} M parâm. | "
              f"{e['tamanho_mb']:>7.1f} MB | {e['img_size']}px | "
              f"{e['latencia_ms_batch1']:>7.2f} ms/img | F1={e['f1_ensemble']:.4f}")
    saida['eficiencia'] = efic

    # ── H3: variância por classe entre folds ─────────────────────────────────
    print(f"\n{'=' * 70}\nH3 — DESVIO PADRÃO DO F1 POR CLASSE ENTRE FOLDS\n{'=' * 70}")
    print(f"  {'modelo':<16} {'família':<8} " + " ".join(f"{c:>7}" for c in CLASSES_H3))
    h3 = {}
    for m in modelos:
        dps = {}
        for c in CLASSES_H3:
            i = LABEL_TO_IDX[c]
            f1s = [f1_score(y, p, labels=[i], average='macro') for p in pred[m]['preds_folds']]
            dps[c] = float(np.std(f1s, ddof=1))
        h3[m] = {'familia': FAMILIA.get(m, '?'), 'dp_f1_por_classe': dps}
        print(f"  {m:<16} {FAMILIA.get(m, '?'):<8} " + " ".join(f"{dps[c]:>7.4f}" for c in CLASSES_H3))
    saida['h3'] = h3

    with open(RESULT_DIR / 'significancia_final.json', 'w', encoding='utf-8') as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    log(f"Concluído. Salvo em: {RESULT_DIR / 'significancia_final.json'}")
    log(f"Log completo em:    {RESULT_DIR / 'significancia_log.txt'}")


if __name__ == "__main__":
    main()
