import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

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

_reader = None  


def get_reader():
    """
    Inicializa o leitor EasyOCR uma única vez.
    gpu=False garante funcionamento mesmo sem GPU/CUDA disponível.
    """
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def clean_text(text: str) -> str:
    """Remove caracteres não alfanuméricos e converte para maiúsculas."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate(text: str) -> bool:
    """
    Valida formato de placa brasileira:
      - Antigo:   AAA9999
      - Mercosul: AAA9A99
    """
    if len(text) != 7:
        return False
    old      = bool(re.match(r'^[A-Z]{3}[0-9]{4}$', text))
    mercosul = bool(re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', text))
    return old or mercosul


def select_best(candidates: list[tuple[str, float]]) -> tuple[str, float]:
    """
    Seleciona o melhor candidato entre (texto, confiança):
      1. Prefere placas válidas com maior confiança.
      2. Fallback: texto de 7 chars com maior confiança.
      3. Último recurso: qualquer texto não-vazio.
    """
    if not candidates:
        return "", 0.0

    valid = [(t, c) for t, c in candidates if is_valid_plate(t)]
    if valid:
        return max(valid, key=lambda x: x[1])

    sevens = [(t, c) for t, c in candidates if len(t) == 7]
    if sevens:
        return max(sevens, key=lambda x: x[1])

    non_empty = [(t, c) for t, c in candidates if t]
    if non_empty:
        return max(non_empty, key=lambda x: x[1])

    return "", 0.0


def merge_fragments(candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
    EasyOCR às vezes divide a placa em fragmentos (ex.: 'ABC' e '1D23').
    Tenta concatenar pares adjacentes para reconstruir a placa completa.
    """
    texts = [t for t, _ in candidates]
    merged = list(candidates)

    for i, (t1, c1) in enumerate(candidates):
        for j, (t2, c2) in enumerate(candidates):
            if i == j:
                continue
            joined = t1 + t2
            if 6 <= len(joined) <= 8 and is_valid_plate(joined[:7]):
                merged.append((joined[:7], (c1 + c2) / 2))

    return merged


def run_easyocr(img_array: np.ndarray,
                min_confidence: float = 0.05) -> list[tuple[str, float]]:
    """
    Executa EasyOCR em um array NumPy (BGR ou cinza).
    Retorna lista de (texto_limpo, confiança) em ordem decrescente de confiança.
    """
    reader = get_reader()

    # EasyOCR funciona melhor com imagens RGB
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    try:
        raw = reader.readtext(
            img_rgb,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
    except Exception:
        return []

    results = []
    for (_bbox, text, conf) in raw:
        cleaned = clean_text(text)
        if cleaned and conf >= min_confidence:
            results.append((cleaned, float(conf)))

    return sorted(results, key=lambda x: x[1], reverse=True)


def process_image(image_path: str, debug_dir: Path | None = None) -> dict:
    """
    Executa o pipeline completo para uma imagem:

      Etapa 1 – Carregamento e redimensionamento.
      Etapa 2 – Detecção da placa por análise de contorno.
      Etapa 3 – Fallback por cor azul (Mercosul) se contorno falhar.
      Etapa 4 – Fallback por varredura de contraste se cor falhar.
      Etapa 5 – Pré-processamento: múltiplas variantes de binarização.
      Etapa 6 – OCR com EasyOCR na imagem colorida + variantes binarizadas.
      Etapa 7 – Fusão de fragmentos e seleção do melhor resultado.

    Retorna dicionário com resultado e metadados do processamento.
    """
    fname = os.path.basename(image_path)
    result = {
        "arquivo": fname,
        "placa_detectada": False,
        "metodo_deteccao": "",
        "texto_lido": "",
        "confianca": 0.0,
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
            # ── Etapa 3: Fallback por cor azul 
            plate_roi = fallback_roi_blue(img)
            if plate_roi is not None and plate_roi.size > 0:
                result["metodo_deteccao"] = "cor_azul"
            else:
                # ── Etapa 4: Fallback por contraste 
                plate_roi = fallback_roi_hc(img)
                result["metodo_deteccao"] = "contraste"

        if debug_dir is not None and plate_roi is not None:
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{Path(fname).stem}_roi.jpg"), plate_roi)

        # ── Etapa 5: Pré-processamento 
        variants = preprocess_standard(plate_roi)

        # ── Etapa 6: OCR em imagem colorida + variantes 
        all_candidates: list[tuple[str, float]] = []

        # Imagem colorida (EasyOCR aproveita informação de cor)
        color_readings = run_easyocr(plate_roi)
        all_candidates.extend(color_readings)

        # Variantes binarizadas
        for _vname, variant_img in variants:
            readings = run_easyocr(variant_img)
            all_candidates.extend(readings)

        # ── Etapa 7: Fusão e seleção 
        all_candidates = merge_fragments(all_candidates)

        # Deduplica mantendo maior confiança por texto
        best_conf: dict[str, float] = {}
        for text, conf in all_candidates:
            if text not in best_conf or conf > best_conf[text]:
                best_conf[text] = conf
        unique = [(t, c) for t, c in best_conf.items()]

        result["candidatos"] = [
            {"texto": t, "confianca": round(c, 4)} for t, c in unique
        ]
        best_text, best_c = select_best(unique)
        result["texto_lido"]  = best_text
        result["confianca"]   = round(best_c, 4)
        result["placa_valida"] = is_valid_plate(best_text)

    except Exception as exc:
        result["erro"] = str(exc)

    return result


def run_pipeline(images_dir: str, output_dir: str, debug: bool = False) -> None:
    """
    Processa todas as imagens em images_dir.
    Gera em output_dir:
      - resultados_easyocr.txt  : tabela legível resumida
      - resultados_easyocr.json : dados completos
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir  = (output_dir / "debug") if debug else None

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
        f"\n{'='*72}\n"
        f"  RESULTADOS – EASYOCR  |  Dataset: Placas Mercosul BR\n"
        f"{'='*72}\n"
        f"{'#':<4} {'Arquivo':<16} {'Placa Lida':<12} "
        f"{'Conf.':<8} {'Válida?':<8} {'Método':<16}\n"
        f"{'-'*72}"
    )
    lines = [header]

    for img_path in image_files:
        print(f"  [{img_path.name}] processando...")
        res = process_image(str(img_path), debug_dir=debug_dir)
        all_results.append(res)

        num    = re.sub(r"\D", "", img_path.stem)
        placa  = res.get("texto_lido", "")
        conf   = res.get("confianca", 0.0)
        valida = "SIM" if res.get("placa_valida") else "NÃO"
        metodo = res.get("metodo_deteccao", "—")
        erro   = f"  ⚠ {res['erro']}" if "erro" in res else ""

        line = (f"{num:<4} {res['arquivo']:<16} {placa:<12} "
                f"{conf:<8.3f} {valida:<8} {metodo:<16}{erro}")
        lines.append(line)

    validas = sum(1 for r in all_results if r.get("placa_valida"))
    footer = (
        f"{'-'*72}\n"
        f"Total: {len(all_results)} imagens  |  "
        f"Leituras no formato válido: {validas}/{len(all_results)}\n"
        f"{'='*72}\n\n"
        f"NOTA: EasyOCR utiliza redes neurais profundas (CRAFT + CRNN),\n"
        f"sendo mais robusto a marcas d'água e variações de iluminação.\n"
        f"{'='*72}"
    )
    lines.append(footer)

    txt_path = output_dir / "resultados_easyocr.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [OK] {txt_path}")

    json_path = output_dir / "resultados_easyocr.json"
    json_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  [OK] {json_path}")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline EasyOCR – Placas Veiculares Brasileiras"
    )
    parser.add_argument("--images", default="./images")
    parser.add_argument("--output", default="./results_easyocr")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print("  PIPELINE EASYOCR – PLACAS VEICULARES")
    print(f"{'='*72}")
    print(f"  Imagens : {args.images}")
    print(f"  Saída   : {args.output}")
    print(f"{'='*72}\n")

    run_pipeline(args.images, args.output, debug=args.debug)
