import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pytesseract
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils.preprocessing import (
    load_image,
    resize_if_needed,
    detect_plate_contour,
    extract_plate_roi,
    fallback_roi_blue,
    fallback_roi_hc,
    preprocess_standard,
)

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TESS_CONFIGS = [
    f"--oem 3 --psm 7 -c tessedit_char_whitelist={WHITELIST}",
    f"--oem 3 --psm 6 -c tessedit_char_whitelist={WHITELIST}",
    f"--oem 3 --psm 8 -c tessedit_char_whitelist={WHITELIST}",
    f"--oem 3 --psm 13 -c tessedit_char_whitelist={WHITELIST}",
]

def clean_text(text: str) -> str:
    """Remove qualquer caractere não alfanumérico e converte para maiúsculas."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate(text: str) -> bool:
    
    if len(text) != 7:
        return False
    old      = bool(re.match(r'^[A-Z]{3}[0-9]{4}$', text))
    mercosul = bool(re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', text))
    return old or mercosul


def select_best(candidates: list[str]) -> str:
    """
    Seleciona o melhor candidato entre múltiplas leituras OCR:
      1. Prefere textos que passam na validação de placa (7 chars no formato correto).
      2. Usa votação por maioria entre resultados de mesmo comprimento.
      3. Fallback: resultado mais longo.
    """
    if not candidates:
        return ""

    # Prioriza placas válidas por votação
    valid = [t for t in candidates if is_valid_plate(t)]
    if valid:
        return Counter(valid).most_common(1)[0][0]

    # Sem placa válida: vota entre resultados de 7 chars
    sevens = [t for t in candidates if len(t) == 7]
    if sevens:
        return Counter(sevens).most_common(1)[0][0]

    # Fallback: vota entre todos os não-vazios
    non_empty = [t for t in candidates if t]
    if non_empty:
        return Counter(non_empty).most_common(1)[0][0]

    return ""


def run_tesseract(img_array: np.ndarray, config: str) -> str:
    """Executa Tesseract num array NumPy e retorna o texto limpo."""
    try:
        pil_img = Image.fromarray(img_array)
        text = pytesseract.image_to_string(pil_img, lang="eng", config=config)
        return clean_text(text)
    except Exception:
        return ""

def process_image(image_path: str, debug_dir: Path | None = None) -> dict:
    """
    Executa o pipeline completo para uma imagem:

      Etapa 1 – Carregamento e redimensionamento.
      Etapa 2 – Detecção da placa por análise de contorno.
      Etapa 3 – Fallback por cor azul (Mercosul) se contorno falhar.
      Etapa 4 – Fallback por varredura de contraste se cor falhar.
      Etapa 5 – Pré-processamento: múltiplas variantes de binarização.
      Etapa 6 – OCR com Tesseract em múltiplas configurações PSM.
      Etapa 7 – Seleção do melhor resultado por votação.

    Retorna dicionário com resultado e metadados do processamento.
    """
    fname = os.path.basename(image_path)
    result = {
        "arquivo": fname,
        "placa_detectada": False,
        "metodo_deteccao": "",
        "texto_lido": "",
        "placa_valida": False,
        "candidatos": [],
    }

    try:
        # ── Etapa 1: Carregamento 
        img = load_image(image_path)
        img = resize_if_needed(img)

        # ── Etapa 2: Detecção por contorno 
        contour = detect_plate_contour(img)
        plate_roi = extract_plate_roi(img, contour)

        if plate_roi is not None and plate_roi.size > 0:
            result["placa_detectada"] = True
            result["metodo_deteccao"] = "contorno"
        else:
            # ── Etapa 3: Fallback por faixa azul Mercosul 
            plate_roi = fallback_roi_blue(img)
            if plate_roi is not None and plate_roi.size > 0:
                result["metodo_deteccao"] = "cor_azul"
            else:
                # ── Etapa 4: Fallback por contraste 
                plate_roi = fallback_roi_hc(img)
                result["metodo_deteccao"] = "contraste"

        # Salva ROI se em modo debug
        if debug_dir is not None and plate_roi is not None:
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{Path(fname).stem}_roi.jpg"), plate_roi)

        # ── Etapa 5: Pré-processamento 
        variants = preprocess_standard(plate_roi)

        if debug_dir is not None:
            for v_name, v_img in variants:
                cv2.imwrite(
                    str(debug_dir / f"{Path(fname).stem}_{v_name}.jpg"), v_img
                )

        # ── Etapas 6 e 7: OCR + seleção
        all_readings: list[str] = []
        for _vname, variant_img in variants:
            for cfg in TESS_CONFIGS:
                reading = run_tesseract(variant_img, cfg)
                if reading:
                    all_readings.append(reading)

        result["candidatos"] = list(set(all_readings))
        best = select_best(all_readings)
        result["texto_lido"] = best
        result["placa_valida"] = is_valid_plate(best)

    except Exception as exc:
        result["erro"] = str(exc)

    return result


def run_pipeline(images_dir: str, output_dir: str, debug: bool = False) -> None:
    """
    Processa todas as imagens em images_dir.
    Gera em output_dir:
      - resultados_tesseract.txt  : tabela legível resumida
      - resultados_tesseract.json : dados completos (candidatos, método, etc.)
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir  = (output_dir / "debug") if debug else None

    # Ordena numericamente
    image_files = sorted(
        [p for p in images_dir.iterdir()
         if p.suffix.upper() in (".JPG", ".JPEG", ".PNG")],
        key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)
    )

    if not image_files:
        print(f"[AVISO] Nenhuma imagem encontrada em {images_dir}")
        return

    all_results: list[dict] = []

    header = (
        f"\n{'='*68}\n"
        f"  RESULTADOS – TESSERACT OCR  |  Dataset: Placas Mercosul BR\n"
        f"{'='*68}\n"
        f"{'#':<4} {'Arquivo':<16} {'Placa Lida':<12} "
        f"{'Válida?':<8} {'Método':<16}\n"
        f"{'-'*68}"
    )
    lines = [header]

    for img_path in image_files:
        print(f"  [{img_path.name}] processando...")
        res = process_image(str(img_path), debug_dir=debug_dir)
        all_results.append(res)

        num     = re.sub(r"\D", "", img_path.stem)
        placa   = res.get("texto_lido", "")
        valida  = "SIM" if res.get("placa_valida") else "NÃO"
        metodo  = res.get("metodo_deteccao", "—")
        erro    = f"  ⚠ {res['erro']}" if "erro" in res else ""

        line = (f"{num:<4} {res['arquivo']:<16} {placa:<12} "
                f"{valida:<8} {metodo:<16}{erro}")
        lines.append(line)

    validas = sum(1 for r in all_results if r.get("placa_valida"))
    footer = (
        f"{'-'*68}\n"
        f"Total: {len(all_results)} imagens  |  "
        f"Leituras no formato válido: {validas}/{len(all_results)}\n"
        f"{'='*68}\n\n"
        f"NOTA: Este dataset possui uma marca d'água digital ('BRASIL MERCOSUL'\n"
        f"repetida em tom médio-cinza) sobreposta à região de caracteres da placa,\n"
        f"o que interfere na binarização e pode reduzir a acurácia do OCR.\n"
        f"{'='*68}"
    )
    lines.append(footer)

    # ── Salva .txt 
    txt_path = output_dir / "resultados_tesseract.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [OK] {txt_path}")

    # ── Salva .json 
    json_path = output_dir / "resultados_tesseract.json"
    json_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  [OK] {json_path}")

    # Exibe no console
    print("\n" + "\n".join(lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Tesseract OCR – Placas Veiculares Brasileiras"
    )
    parser.add_argument("--images", default="./images",
                        help="Diretório com as imagens (padrão: ./images)")
    parser.add_argument("--output", default="./results_tesseract",
                        help="Diretório de saída (padrão: ./results_tesseract)")
    parser.add_argument("--debug", action="store_true",
                        help="Salva ROIs e variantes de pré-processamento")
    args = parser.parse_args()

    print(f"\n{'='*68}")
    print("  PIPELINE TESSERACT OCR – PLACAS VEICULARES")
    print(f"{'='*68}")
    print(f"  Imagens : {args.images}")
    print(f"  Saída   : {args.output}")
    print(f"  Debug   : {args.debug}")
    print(f"{'='*68}\n")

    run_pipeline(args.images, args.output, debug=args.debug)
