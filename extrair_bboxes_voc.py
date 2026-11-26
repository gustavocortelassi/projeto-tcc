"""
Extrai cada objeto anotado (Pascal VOC XML) da imagem-fonte, recortando
pelo bounding box, e salva em pasta por classe.

Filtra dois casos que produzem recorte sem estrutura reconhecível:
  - Caixa menor que MIN_LADO em qualquer dimensão (ex.: 42x52px ampliado
    pra 224x224 vira mancha sem forma identificável).
  - Objeto marcado truncated=1 (planta cortada na borda da foto — visão
    parcial, mesmo problema de fundo).

Não deleta nada — os que foram filtrados ficam registrados em
objetos_filtrados.json, com o motivo, para auditoria.

Uso:
    python extrair_bboxes_voc.py
"""

import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── Ajuste antes de rodar ───────────────────────────────────────────────────
PASTAS_ORIGEM = [Path("train_set"), Path("validation_set"), Path("test_set")]
DESTINO = Path("Dataset V1 a V4 Externo")

MIN_LADO = 100          # px — abaixo disso, descarta
EXCLUIR_TRUNCADO = True  # descarta objetos com truncated=1
CLASSES_DESEJADAS = {'V1', 'V3', 'V4'}  # só milho — descarta erva daninha (CYPRO/ECHCG/SOLNI) e o resto


def extrair_classe(nome_completo):
    # formato observado: "ZEAMX_V1" -> "V1"
    if nome_completo.startswith('ZEAMX_'):
        return nome_completo[len('ZEAMX_'):]
    return nome_completo


def processar_xml(xml_path, pasta_split):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nome_imagem = root.findtext('filename')
    img_path = xml_path.parent / nome_imagem
    if not img_path.exists():
        # extensão pode divergir de maiúscula/minúscula
        candidatos = list(xml_path.parent.glob(f"{xml_path.stem}.*"))
        candidatos = [c for c in candidatos if c.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if not candidatos:
            return [], [{'arquivo': str(xml_path), 'motivo': 'imagem-fonte não encontrada'}]
        img_path = candidatos[0]

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return [], [{'arquivo': str(img_path), 'motivo': f'erro ao abrir: {e}'}]

    mantidos = []
    filtrados = []

    for i, obj in enumerate(root.findall('object')):
        nome_classe = extrair_classe(obj.findtext('name', '').strip())
        truncado = obj.findtext('truncated', '0').strip() == '1'
        bnd = obj.find('bndbox')
        xmin = int(float(bnd.findtext('xmin')))
        ymin = int(float(bnd.findtext('ymin')))
        xmax = int(float(bnd.findtext('xmax')))
        ymax = int(float(bnd.findtext('ymax')))

        largura = xmax - xmin
        altura = ymax - ymin

        motivo = None
        if nome_classe not in CLASSES_DESEJADAS:
            motivo = f'espécie/classe fora do escopo ({nome_classe})'
        elif EXCLUIR_TRUNCADO and truncado:
            motivo = 'truncado'
        elif largura < MIN_LADO or altura < MIN_LADO:
            motivo = f'caixa pequena ({largura}x{altura}px)'

        info_base = {
            'xml': str(xml_path), 'objeto_idx': i, 'classe': nome_classe,
            'bbox': [xmin, ymin, xmax, ymax], 'largura': largura, 'altura': altura,
        }

        if motivo:
            filtrados.append({**info_base, 'motivo': motivo})
            continue

        recorte = img.crop((xmin, ymin, xmax, ymax))
        nome_arquivo = f"{pasta_split.name}__{xml_path.stem}__obj{i:02d}.jpg"
        mantidos.append({'classe': nome_classe, 'imagem': recorte, 'nome': nome_arquivo})

    return mantidos, filtrados


def main():
    todos_filtrados = []
    contagem_por_classe = {}
    total_processados = 0

    for pasta_split in PASTAS_ORIGEM:
        if not pasta_split.exists():
            print(f"[AVISO] Pasta não encontrada, pulando: {pasta_split}")
            continue

        print(f"Processando: {pasta_split}")
        xmls = list(pasta_split.glob('*.xml'))

        for xml_path in xmls:
            mantidos, filtrados = processar_xml(xml_path, pasta_split)
            todos_filtrados += filtrados

            for item in mantidos:
                destino_dir = DESTINO / item['classe']
                destino_dir.mkdir(parents=True, exist_ok=True)
                destino = destino_dir / item['nome']
                if not destino.exists():
                    item['imagem'].save(destino, quality=95)
                contagem_por_classe[item['classe']] = contagem_por_classe.get(item['classe'], 0) + 1
                total_processados += 1

    print(f"\n{'=' * 60}\nRESUMO\n{'=' * 60}")
    for classe, n in sorted(contagem_por_classe.items()):
        print(f"  {classe:<10} {n}")
    print(f"\nTotal extraído: {total_processados}")
    print(f"Total filtrado (truncado ou caixa pequena): {len(todos_filtrados)}")

    motivos = {}
    especies_descartadas = {}
    for f in todos_filtrados:
        chave = f['motivo'].split(' (')[0]
        motivos[chave] = motivos.get(chave, 0) + 1
        if 'fora do escopo' in f['motivo']:
            especies_descartadas[f['classe']] = especies_descartadas.get(f['classe'], 0) + 1
    for motivo, n in motivos.items():
        print(f"  {motivo}: {n}")

    if especies_descartadas:
        print(f"\nDescartado por espécie/classe fora do escopo, detalhado:")
        for especie, n in sorted(especies_descartadas.items(), key=lambda x: -x[1]):
            print(f"  {especie:<12} {n}")

    with open('objetos_filtrados.json', 'w', encoding='utf-8') as f:
        json.dump(todos_filtrados, f, indent=2, ensure_ascii=False)
    print(f"\nDetalhe dos filtrados em: objetos_filtrados.json")
    print(f"Extraído em: {DESTINO}")
    print("Próximo passo: python montar_dataset_final.py")


if __name__ == "__main__":
    main()