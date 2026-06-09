import os, re, sys, json, argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\Nova pasta\tesseract.exe'
from PIL import Image


# ─── DETECÇÃO DA PLACA ──────────────────────────────────────────────────────

def detect_plate(img: np.ndarray):
    """
    Detecta e recorta a região da placa usando três estratégias encadeadas:

    1. Borda escura (dark frame)
       A placa sobreposta tem uma moldura de pixels muito escuros (<15).
       Dilata a máscara escura e busca o maior retângulo com aspecto entre 2 e 8.

    2. Faixa azul Mercosul
       Encontra o maior componente azul (hue 90-130) horizontal e extrai
       o corpo da placa logo abaixo.

    3. Varredura de alto contraste
       Localiza a faixa de linhas com maior desvio-padrão horizontal
       (os caracteres criam transições branco↔preto).
    """
    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, dark = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 8))
    dark_d = cv2.dilate(dark, k, iterations=2)
    cnts, _ = cv2.findContours(dark_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        x, y, w, h = cv2.boundingRect(cnt)
        if 2.0 < w / max(h, 1) < 8.0 and 0.02 < (w * h) / (iw * ih) < 0.6:
            y1, y2 = max(0, y - 5), min(ih, y + h + 5)
            x1, x2 = max(0, x - 5), min(iw, x + w + 5)
            roi = img[y1:y2, x1:x2]
            if roi.shape[0] > 20 and roi.shape[1] > 60:
                return roi, "moldura_escura"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k2)
    n, _, stats, _ = cv2.connectedComponentsWithStats(blue)
    cands = []
    for i in range(1, n):
        x, y, w, h = [int(v) for v in stats[i][:4]]
        if w < 60 or not (5 <= h <= 100):
            continue
        roi_h = hsv[y:y + h, x:x + w, 0]
        bf = ((roi_h >= 90) & (roi_h <= 130)).mean()
        if bf > 0.15 and w / max(h, 1) > 2.0:
            cands.append((stats[i][4] * bf, x, y, w, h))
    if cands:
        _, bx, by, bw, bh = max(cands)
        y1, y2 = by, min(ih, by + int(bh * 6))
        x1, x2 = max(0, bx - 5), min(iw, bx + bw + 5)
        roi = img[y1:y2, x1:x2]
        if roi.shape[0] > 20 and roi.shape[1] > 60:
            return roi, "faixa_azul"

    row_stds = np.array([gray[r, iw // 6:5 * iw // 6].std() for r in range(ih)])
    sm = np.convolve(row_stds, np.ones(5) / 5, mode="same")
    bs = be = cs = 0; ib = False
    for r in range(ih):
        if sm[r] > 40:
            if not ib: cs = r; ib = True
        elif ib:
            if r - cs > be - bs: bs, be = cs, r
            ib = False
    if ib and ih - cs > be - bs:
        bs, be = cs, ih
    if be - bs > 20:
        band = gray[bs:be, :]
        hc = np.where(band.std(axis=0) > 25)[0]
        if len(hc) > 30:
            roi = img[max(0, bs):min(ih, be),
                      max(0, int(hc[0]) - 5):min(iw, int(hc[-1]) + 5)]
            if roi.shape[0] > 10 and roi.shape[1] > 30:
                return roi, "alto_contraste"

    # Fallback
    return img[ih // 3:2 * ih // 3, iw // 8:7 * iw // 8], "fallback"


def preprocess(roi: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Gera múltiplas variantes binarizadas para maximizar acerto do OCR.

    Desafio: o dataset possui marca d'água ('BRASIL MERCOSUL' repetida em
    tom médio-cinza ~80-150) sobreposta ao fundo branco (>200) da placa.
    Os caracteres reais são muito escuros (<30).

    Técnicas aplicadas:
    - Threshold adaptativo (principal): usa janela local → reduz efeito do watermark.
    - Threshold fixo em valores acima de 60 (remove parte do watermark).
    - CLAHE + Otsu: equalização local de histograma antes de binarizar.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    h, w = gray.shape

    # Remove faixa azul (topo ~20%) e margem lateral (3%)
    skip_top = max(0, int(h * 0.20))
    xt = max(0, int(w * 0.03))
    char = gray[skip_top:, xt:w - xt]
    if char.shape[0] < 5 or char.shape[1] < 10:
        char = gray

    ch, cw = char.shape
    if ch < 60:
        scale = max(2, int(60 / max(ch, 1)))
        char = cv2.resize(char, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)

    variants = []

    for bs, C in [(31, 10), (41, 12), (51, 14), (61, 16)]:
        try:
            ad = cv2.adaptiveThreshold(char, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, bs, C)
            variants.append((f"adapt_{bs}_{C}", ad))
            variants.append((f"adapt_{bs}_{C}_inv", cv2.bitwise_not(ad)))
        except Exception:
            pass

    for th in [60, 75, 90, 105]:
        try:
            _, b = cv2.threshold(char, th, 255, cv2.THRESH_BINARY)
            variants.append((f"thresh_{th}", b))
        except Exception:
            pass

    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(char)
        _, otsu = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("clahe_otsu", otsu))
        variants.append(("clahe_otsu_inv", cv2.bitwise_not(otsu)))
    except Exception:
        pass

    return variants


# ─── OCR ────────────────────────────────────────────────────────────────────

WL  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CFG7 = f"--oem 3 --psm 7 -c tessedit_char_whitelist={WL}"
CFG6 = f"--oem 3 --psm 6 -c tessedit_char_whitelist={WL}"
CFG8 = f"--oem 3 --psm 8 -c tessedit_char_whitelist={WL}"


def run_ocr(img_array: np.ndarray) -> list[str]:
    """Executa Tesseract com múltiplos PSM e retorna lista de leituras."""
    results = []
    pil = Image.fromarray(img_array)
    for cfg in [CFG7, CFG6, CFG8]:
        try:
            t = pytesseract.image_to_string(pil, lang="eng", config=cfg)
            t = re.sub(r"[^A-Z0-9]", "", t.upper())
            if 4 <= len(t) <= 10:
                results.append(t)
        except Exception:
            pass
    return results


def clean(t: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", t.upper())


def is_valid(t: str) -> bool:
    if len(t) != 7:
        return False
    return bool(re.match(r"^[A-Z]{3}[0-9]{4}$", t)) or \
           bool(re.match(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$", t))


def best_reading(reads: list[str]) -> str:
    sevens = [t for t in reads if len(t) == 7]
    if sevens:
        return Counter(sevens).most_common(1)[0][0]
    if reads:
        return Counter(reads).most_common(1)[0][0]
    return ""


# ─── PROCESSAMENTO POR IMAGEM ────────────────────────────────────────────────

def process(path: str, save_roi_dir: Path | None = None) -> dict:
    """Pipeline completo para uma imagem."""
    fname = os.path.basename(path)
    result = dict(arquivo=fname, metodo="", texto="", valida=False, candidatos=[])
    try:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)

        # 1. Detecção
        roi, method = detect_plate(img)
        result["metodo"] = method

        if save_roi_dir:
            save_roi_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_roi_dir / f"{Path(fname).stem}_roi.jpg"), roi)

        # 2. Pré-processamento
        variants = preprocess(roi)

        # 3. OCR
        all_reads: list[str] = []
        for _name, vimg in variants:
            all_reads.extend(run_ocr(vimg))

        result["candidatos"] = list(set(all_reads))
        best = best_reading(all_reads)
        result["texto"] = best
        result["valida"] = is_valid(best)
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
        "\n" + "=" * 70,
        "  RESULTADOS – TESSERACT OCR  |  Placas Mercosul BR",
        "=" * 70,
        f"{'#':<4} {'Arquivo':<16} {'Placa Lida':<12} {'Válida?':<8} {'Método'}",
        "-" * 70,
    ]

    for fp in files:
        num = re.sub(r"\D", "", fp.stem)
        print(f"  [{fp.name}] processando...")
        r = process(str(fp), save_roi_dir=dbg)
        all_res.append(r)
        rows.append(
            f"{num:<4} {r['arquivo']:<16} {r['texto']:<12} "
            f"{'SIM' if r['valida'] else 'NÃO':<8} {r['metodo']}"
            + (f"  ⚠ {r['erro']}" if "erro" in r else "")
        )

    validas = sum(1 for r in all_res if r["valida"])
    rows += [
        "-" * 70,
        f"Total: {len(all_res)} imagens  |  Válidas: {validas}/{len(all_res)}",
        "=" * 70,
        "",
        "OBSERVAÇÃO: As imagens possuem marca d'água digital ('BRASIL MERCOSUL'",
        "repetida em tom médio-cinza) sobreposta aos caracteres da placa.",
        "Isso interfere na binarização e pode reduzir a acurácia do OCR.",
        "=" * 70,
    ]

    txt = out / "resultados_tesseract.txt"
    txt.write_text("\n".join(rows), encoding="utf-8")
    (out / "resultados_tesseract.json").write_text(
        json.dumps(all_res, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Resultados salvos em: {out}")
    print("\n" + "\n".join(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="./images")
    ap.add_argument("--output", default="./results_tesseract")
    ap.add_argument("--debug",  action="store_true")
    args = ap.parse_args()
    run(args.images, args.output, args.debug)
