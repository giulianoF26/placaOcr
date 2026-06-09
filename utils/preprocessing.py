import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Carrega imagem em BGR; levanta FileNotFoundError se o arquivo não existir."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    return img


def resize_if_needed(img: np.ndarray, max_width: int = 1400) -> np.ndarray:
    """Redimensiona a imagem para no máximo max_width pixels de largura,
    mantendo a proporção. Imagens menores não são alteradas."""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def detect_plate_contour(img: np.ndarray):
    """
    Detecta o contorno retangular da placa por análise de bordas.

    Pipeline:
      1. Converte para escala de cinza e aplica filtro bilateral
         (reduz ruído preservando arestas).
      2. Detecta bordas com Canny.
      3. Fechamento morfológico para unir bordas próximas.
      4. Filtra contornos pelo maior quadrilátero com razão de aspecto
         próxima a de uma placa Mercosul (~4.5 : 1).

    Retorna o contorno (4 pontos) ou None se não encontrado.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Filtro bilateral: preserva bordas enquanto suaviza regiões homogêneas
    blur = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)

    # Detecção de bordas com Canny
    edges = cv2.Canny(blur, 30, 200)

    # Fechamento morfológico para unir bordas próximas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Ordena contornos do maior para o menor
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    h_img, w_img = img.shape[:2]

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h if h > 0 else 0
            area_ratio = (w * h) / (w_img * h_img)

            # Placa Mercosul: razão de aspecto ~2 a 6, área relevante
            if 1.8 < aspect < 6.5 and 0.005 < area_ratio < 0.45:
                return approx

    return None


def extract_plate_roi(img: np.ndarray, contour) -> np.ndarray | None:
    """
    Extrai e retifica a região da placa usando transformação de perspectiva.
    Aplica correção de perspectiva (warp) para lidar com ângulos de câmera.

    Retorna a imagem recortada da placa ou None se o contorno for inválido.
    """
    if contour is None:
        return None

    pts = contour.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width  = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    if width < 50 or height < 10:
        return None

    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos: topo-esq, topo-dir, baixo-dir, baixo-esq."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def fallback_roi_blue(img: np.ndarray) -> np.ndarray:
    """
    Estratégia alternativa: segmenta pela faixa azul do padrão Mercosul.
    A faixa BRASIL está no topo da placa; o corpo (branco com caracteres)
    fica diretamente abaixo.

    Nota: este dataset possui uma marca d'água (watermark) sobreposta
    com o texto 'BRASIL MERCOSUL' em tom médio-cinza que interfere com
    a detecção por cor.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ih, iw = img.shape[:2]

    # Máscara para a faixa azul do padrão Mercosul
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Fecha pequenos buracos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    best = None
    best_score = 0
    for i in range(1, n):
        x, y, w, h = [int(v) for v in stats[i][:4]]
        area = stats[i][4]
        aspect = w / h if h > 0 else 0
        if w < 60 or h < 5 or h > 100:
            continue
        # Verifica densidade real de azul na região
        roi_h = hsv[y:y+h, x:x+w, 0]
        blue_frac = ((roi_h >= 90) & (roi_h <= 130)).mean()
        score = area * blue_frac
        if score > best_score and aspect > 2.0:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        # Fallback final: porção central inferior da imagem
        return img[ih // 3: 2 * ih // 3, iw // 8: 7 * iw // 8]

    bx, by, bw, bh = best
    # Corpo da placa: logo abaixo da faixa azul
    y1 = by + bh
    y2 = min(ih, by + int(bh * 5.5))
    x1 = max(0, bx - 5)
    x2 = min(iw, bx + bw + 5)

    roi = img[y1:y2, x1:x2]
    return roi if roi.size > 0 else img[ih // 3: 2 * ih // 3, iw // 8: 7 * iw // 8]


def fallback_roi_hc(img: np.ndarray) -> np.ndarray:
    """
    Estratégia por varredura de contraste:
    Encontra a faixa de linhas com maior desvio padrão horizontal
    (caracteres na placa criam alto contraste vertical).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape

    row_stds = np.array([gray[r, iw // 6: 5 * iw // 6].std()
                         for r in range(ih)])
    sm = np.convolve(row_stds, np.ones(5) / 5, mode='same')

    best_s, best_e = 0, 0
    cur_s = 0
    in_band = False
    for r in range(ih):
        if sm[r] > 35:
            if not in_band:
                cur_s = r
                in_band = True
        else:
            if in_band:
                if r - cur_s > best_e - best_s:
                    best_s, best_e = cur_s, r
                in_band = False
    if in_band and ih - cur_s > best_e - best_s:
        best_s, best_e = cur_s, ih

    if best_e - best_s > 15:
        band = gray[best_s:best_e, :]
        col_stds = band.std(axis=0)
        hc = np.where(col_stds > 25)[0]
        if len(hc) > 30:
            y1 = max(0, best_s)
            y2 = min(ih, best_e)
            x1 = max(0, int(hc[0]) - 5)
            x2 = min(iw, int(hc[-1]) + 5)
            return img[y1:y2, x1:x2]

    return img[ih // 3: 2 * ih // 3, iw // 8: 7 * iw // 8]


def preprocess_standard(plate_img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Gera múltiplas variantes de pré-processamento otimizadas para este dataset.

    Desafio específico: este dataset possui marca d'água (watermark) em tom
    médio-cinza (~80-150) sobreposta à placa branca (>200) com caracteres
    pretos (<30). As estratégias abaixo tentam separar os caracteres reais
    do ruído da marca d'água.

    Retorna lista de (nome, imagem_processada) para passar ao OCR.
    """
    if plate_img is None or plate_img.size == 0:
        return []

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) \
        if len(plate_img.shape) == 3 else plate_img.copy()

    # Escalonamento: garante altura mínima de 80px para OCR preciso
    h, w = gray.shape
    scale = max(2, min(6, int(120 / max(h, 1))))
    gray_up = cv2.resize(gray, (w * scale, h * scale),
                         interpolation=cv2.INTER_CUBIC)

    variants = []

    # --- Variante 1: Limiar fixo em 80 (separa chars pretos do watermark cinza) ---
    # Chars reais: <30; watermark: 80-150; fundo: >180
    # Limiar 80 captura chars sem capturar watermark pesado
    _, thresh80 = cv2.threshold(gray_up, 80, 255, cv2.THRESH_BINARY)
    variants.append(("thresh80", thresh80))
    variants.append(("thresh80_inv", cv2.bitwise_not(thresh80)))

    # --- Variante 2: CLAHE + Otsu (equalização adaptativa de histograma) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_up)
    _, otsu = cv2.threshold(enhanced, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu", otsu))
    variants.append(("clahe_otsu_inv", cv2.bitwise_not(otsu)))

    # --- Variante 3: Threshold adaptativo (melhor para iluminação não uniforme) ---
    try:
        adaptive = cv2.adaptiveThreshold(gray_up, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 31, 12)
        variants.append(("adaptive_31_12", adaptive))
        variants.append(("adaptive_31_12_inv", cv2.bitwise_not(adaptive)))
    except Exception:
        pass

    # --- Variante 4: Filtragem por tamanho de blob (remove watermark pequeno) ---
    # O watermark tem caracteres pequenos; os chars da placa são grandes.
    # Binariza e mantém apenas blobs com área >= limiar.
    _, dark = cv2.threshold(gray_up, 80, 255, cv2.THRESH_BINARY_INV)
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats(dark)
    filtered = np.zeros_like(dark)
    min_blob = max(50, h * scale * 3)  # escala com o tamanho da imagem
    for j in range(1, n):
        blob_h = stats_cc[j][3]
        blob_area = stats_cc[j][4]
        # Mantém blobs altos (caracteres reais têm altura proporcional à placa)
        if blob_area >= min_blob and blob_h > h * scale * 0.25:
            filtered[labels == j] = 255
    blob_result = cv2.bitwise_not(filtered)
    variants.append(("blob_filter", blob_result))

    return variants
