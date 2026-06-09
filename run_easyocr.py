import os, re, sys, json, argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Reutiliza as mesmas funções de detecção do pipeline Tesseract
from run_tesseract import detect_plate, preprocess, is_valid


_reader = None  


def get_reader():
    
    global _reader
    if _reader is None:
        import easyocr  # import tardio – evita erro se não instalado
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def run_easyocr_on(img_array: np.ndarray,
                   min_conf: float = 0.05) -> list[tuple[str, float]]:
    """
    Executa EasyOCR em um array NumPy (BGR ou cinza).

    EasyOCR usa dois estágios:
      1. CRAFT  – detector de regiões de texto.
      2. CRNN   – reconhecedor baseado em rede recorrente.

    Diferença em relação ao Tesseract:
      - Mais robusto a variações de fonte e perspectiva.
      - Retorna confiança por leitura.
      - Pode ser mais lento sem GPU.

    Retorna lista de (texto_limpo, confiança), decrescente por confiança.
    """
    reader = get_reader()

    # EasyOCR trabalha com RGB
    if len(img_array.shape) == 2:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    try:
        raw = reader.readtext(
            rgb,
            detail=1,
            paragraph=False,
            allowlist=WL,
        )
    except Exception:
        return []

    results = []
    for (_bbox, text, conf) in raw:
        t = re.sub(r"[^A-Z0-9]", "", text.upper())
        if t and conf >= min_conf:
            results.append((t, float(conf)))

    return sorted(results, key=lambda x: x[1], reverse=True)


def merge_frags(cands: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
    Tenta reconstruir a placa concatenando pares de fragmentos.
    EasyOCR às vezes divide o texto em múltiplos blocos.
    """
    merged = list(cands)
    for i, (t1, c1) in enumerate(cands):
        for j, (t2, c2) in enumerate(cands):
            if i == j:
                continue
            joined = t1 + t2
            if 6 <= len(joined) <= 8 and is_valid(joined[:7]):
                merged.append((joined[:7], (c1 + c2) / 2))
    return merged


def best_easyocr(cands: list[tuple[str, float]]) -> tuple[str, float]:
    """Seleciona o melhor candidato por validade e confiança."""
    if not cands:
        return "", 0.0
    valid = [(t, c) for t, c in cands if is_valid(t)]
    if valid:
        return max(valid, key=lambda x: x[1])
    sevens = [(t, c) for t, c in cands if len(t) == 7]
    if sevens:
        return max(sevens, key=lambda x: x[1])
    non_empty = [(t, c) for t, c in cands if t]
    return max(non_empty, key=lambda x: x[1]) if non_empty else ("", 0.0)


# ─── PROCESSAMENTO POR IMAGEM ────────────────────────────────────────────────

def process(path: str, save_roi_dir: Path | None = None) -> dict:
    """
    Pipeline completo para uma imagem usando EasyOCR:
      1. Detecção da placa (idêntica ao pipeline Tesseract).
      2. OCR na ROI colorida (EasyOCR aproveita a informação de cor).
      3. OCR nas variantes binarizadas (reforço para texto com watermark).
      4. Fusão de fragmentos + seleção por confiança.
    """
    fname = os.path.basename(path)
    result = dict(arquivo=fname, metodo="", texto="", confianca=0.0,
                  valida=False, candidatos=[])
    try:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)

        # Detecção (compartilhada)
        roi, method = detect_plate(img)
        result["metodo"] = method

        if save_roi_dir:
            save_roi_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_roi_dir / f"{Path(fname).stem}_roi.jpg"), roi)

        # OCR na imagem colorida (principal)
        all_cands: list[tuple[str, float]] = []
        all_cands.extend(run_easyocr_on(roi))

        # OCR nas variantes binarizadas
        for _name, vimg in preprocess(roi):
            all_cands.extend(run_easyocr_on(vimg))

        # Mescla fragmentos
        all_cands = merge_frags(all_cands)

        # Deduplica mantendo maior confiança
        best_conf: dict[str, float] = {}
        for t, c in all_cands:
            if t not in best_conf or c > best_conf[t]:
                best_conf[t] = c
        unique = list(best_conf.items())

        result["candidatos"] = [{"texto": t, "confianca": round(c, 4)}
                                 for t, c in unique]
        best_t, best_c = best_easyocr(unique)
        result["texto"]     = best_t
        result["confianca"] = round(best_c, 4)
        result["valida"]    = is_valid(best_t)

    except Exception as e:
        result["erro"] = str(e)
    return result


# ─── PIPELINE EM LOTE ────────────────────────────────────────────────────────

def run(images_dir: str, output_dir: str, debug: bool = False) -> None:
    imgs = Path(images_dir)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dbg  = out / "debug" if debug else None

    files = sorted(
        [p for p in imgs.iterdir() if p.suffix.upper() in (".JPG", ".JPEG", ".PNG")],
        key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)
    )
    if not files:
        print(f"Nenhuma imagem em {imgs}")
        return

    all_res = []
    rows = [
        "\n" + "=" * 74,
        "  RESULTADOS – EASYOCR  |  Placas Mercosul BR",
        "=" * 74,
        f"{'#':<4} {'Arquivo':<16} {'Placa Lida':<12} "
        f"{'Conf.':<8} {'Válida?':<8} {'Método'}",
        "-" * 74,
    ]

    for fp in files:
        num = re.sub(r"\D", "", fp.stem)
        print(f"  [{fp.name}] processando...")
        r = process(str(fp), save_roi_dir=dbg)
        all_res.append(r)
        rows.append(
            f"{num:<4} {r['arquivo']:<16} {r['texto']:<12} "
            f"{r['confianca']:<8.3f} {'SIM' if r['valida'] else 'NÃO':<8} {r['metodo']}"
            + (f"  ⚠ {r['erro']}" if "erro" in r else "")
        )

    validas = sum(1 for r in all_res if r["valida"])
    rows += [
        "-" * 74,
        f"Total: {len(all_res)} imagens  |  Válidas: {validas}/{len(all_res)}",
        "=" * 74,
        "",
        "EasyOCR usa CRAFT (detector) + CRNN (reconhecedor neural).",
        "É mais robusto a distorções, variações de fonte e marcas d'água.",
        "Requer PyTorch. GPU acelera significativamente o processamento.",
        "=" * 74,
    ]

    txt = out / "resultados_easyocr.txt"
    txt.write_text("\n".join(rows), encoding="utf-8")
    (out / "resultados_easyocr.json").write_text(
        json.dumps(all_res, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Resultados salvos em: {out}")
    print("\n" + "\n".join(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="./images")
    ap.add_argument("--output", default="./results_easyocr")
    ap.add_argument("--debug",  action="store_true")
    args = ap.parse_args()
    run(args.images, args.output, args.debug)
