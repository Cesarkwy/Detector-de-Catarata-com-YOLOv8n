#!/usr/bin/env python3
"""
@file infer_video_yolov8_bbox.py
@brief Realiza inferência de detecção de catarata em vídeos usando YOLOv8 com registro de métricas.

@details
Este script processa vídeos para:
- Detectar estruturas do olho: córnea, pupila e catarata usando YOLOv8
- Usar fallback com transformada de Hough para detecção de pupila
- Calcular ângulo dominante da incisão dentro da bounding box da catarata (0-180°, sendo 0° horizontal)
- Salvar vídeo anotado com overlays de detecções
- Exportar métricas por frame em arquivo CSV (área, confiança, distância, ângulo)


Cores no vídeo de saída:
- Vermelho: Córnea (detecção do modelo)
- Azul: Pupila (detecção do modelo)
- Amarelo: Pupila (fallback Hough)
- Verde: Catarata (detecção do modelo)
"""
import argparse
import csv
import math
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------
# Funções Utilitárias
# -----------------------

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    @brief Calcula a distância euclidiana entre dois pontos 2D.
    @param a Tupla contendo coordenadas (x, y) do primeiro ponto.
    @param b Tupla contendo coordenadas (x, y) do segundo ponto.
    @return Distância euclidiana entre os pontos a e b.
    """
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))

def ema_update(prev: Optional[float], new: Optional[float], alpha: float) -> Optional[float]:
    """
    @brief Atualiza um valor usando filtro de média móvel exponencial (EMA).
    @details Útil para suavizar flutuações em métricas entre frames consecutivos.
    @param prev Valor anterior filtrado (pode ser None).
    @param new Novo valor a ser integrado (pode ser None ou NaN).
    @param alpha Fator de suavização (0-1), onde 1 = sem filtragem.
    @return Valor atualizado pelo filtro EMA, ou None se ambos prev e new forem None.
    """
    if new is None or (isinstance(new, float) and math.isnan(new)):
        return prev
    if prev is None:
        return float(new)
    return (1 - alpha) * prev + alpha * float(new)

def match_class_id(names: dict, aliases_csv: str) -> Optional[int]:
    """
    @brief Encontra o ID de classe do modelo YOLOv8 que corresponde aos aliases fornecidos.
    @details Realiza busca case-insensitive parcial entre aliases e nomes de classes.
    @param names Dicionário do modelo contendo mapeamento de ID -> nome da classe.
    @param aliases_csv String com aliases separados por vírgula (ex: "cornea,iris,pupil").
    @return ID da classe encontrada, ou None se nenhum alias corresponder.
    """
    aliases = [s.strip().lower() for s in aliases_csv.split(",") if s.strip()]
    for cid, cname in names.items():
        ln = str(cname).lower()
        if any(alias in ln for alias in aliases):
            return cid
    return None

def clip_int(v: int, lo: int, hi: int) -> int:
    """
    @brief Limita um valor inteiro a um intervalo [lo, hi].
    @param v Valor a ser limitado.
    @param lo Limite inferior (inclusivo).
    @param hi Limite superior (inclusivo).
    @return Valor v limitado ao intervalo.
    """
    return max(lo, min(hi, v))

def find_pupil_hough_in_roi(frame_bgr: np.ndarray, cornea_xyxy: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    @brief Detecta a pupila usando transformada de Hough (detecção de círculo) dentro da ROI da córnea.
    @details Fallback para quando o modelo YOLOv8 não detecta a pupila.
             Usa detecção de círculos da transformada de Hough em escala de cinza.
    @param frame_bgr Frame de entrada em formato BGR (OpenCV).
    @param cornea_xyxy Bounding box da córnea [x1, y1, x2, y2].
    @return Tupla (cx, cy, raio) da pupila detectada, ou None se não encontrada.
    @note Retorna coordenadas no sistema de referência da imagem completa, não da ROI.
    """
    if cornea_xyxy is None or len(cornea_xyxy) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in cornea_xyxy]
    H, W = frame_bgr.shape[:2]
    x1, y1 = clip_int(x1, 0, W - 1), clip_int(y1, 0, H - 1)
    x2, y2 = clip_int(x2, 0, W - 1), clip_int(y2, 0, H - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    minR = max(5, int(0.03 * min(gray.shape[:2])))
    maxR = max(10, int(0.5 * min(gray.shape[:2])))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(0.25 * min(gray.shape[:2])),
        param1=80, param2=18, minRadius=minR, maxRadius=maxR
    )
    if circles is None:
        return None
    c = circles[0][0]
    cx_roi, cy_roi, r = float(c[0]), float(c[1]), float(c[2])
    return (cx_roi + x1, cy_roi + y1, r)

def compute_dominant_orientation_deg(roi_bgr: np.ndarray) -> Optional[float]:
    """
    @brief Calcula o ângulo dominante das linhas dentro de uma ROI (região da imagem).
    @details Detecta linhas usando transformada de Hough e calcula a orientação dominante
             ponderada pelo comprimento das linhas detectadas.
             Útil para determinar a inclinação de uma incisão cirúrgica dentro da catarata.
    @param roi_bgr Região de interesse em formato BGR.
    @return Ângulo dominante em graus [0, 180), onde 0° é horizontal.
            Retorna None se nenhuma linha for detectada.
    @note A orientação é calculada usando média circular ponderada para tratar o intervalo [0,180).
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # equalizar / realocar contraste um pouco
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # deteccao de arestas
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # dilatar ligeiramente para fortalecer linhas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    # Transformada de Hough com segmentos de linha
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=30, minLineLength=max(10, min(roi_bgr.shape[:2])//10), maxLineGap=8)
    if lines is None or len(lines) == 0:
        return None
    angles = []
    weights = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 5:
            continue
        ang = math.degrees(math.atan2(dy, dx))  # -180..180
        # mapear para [0,180)
        if ang < 0:
            ang += 180.0
        # converter angulo tal que 0 eh horizontal (ja esta)
        angles.append(ang)
        weights.append(length)
    if len(angles) == 0:
        return None
    # media circular ponderada em [0,180) (mas como 0==180, converter usando o truque do angulo duplo)
    # converter para radianos
    angles_rad = np.deg2rad(np.array(angles))
    double_angles = 2.0 * angles_rad
    w = np.array(weights)
    sin_sum = np.sum(w * np.sin(double_angles))
    cos_sum = np.sum(w * np.cos(double_angles))
    if cos_sum == 0 and sin_sum == 0:
        return None
    mean_double = math.atan2(sin_sum, cos_sum)
    mean_angle_rad = mean_double / 2.0
    mean_angle_deg = math.degrees(mean_angle_rad)
    if mean_angle_deg < 0:
        mean_angle_deg += 180.0
    # normalizar para [0,180)
    mean_angle_deg = mean_angle_deg % 180.0
    return float(mean_angle_deg)

def draw_dominant_line_on_overlay(overlay, bbox_xy, angle_deg, color=(0,200,0), thickness=2):
    """
    @brief Desenha uma linha representando a orientação dominante sobreposta ao vídeo.
    @details Desenha uma linha centralizada na bounding box, com comprimento proporcional
             ao tamanho da caixa e ângulo conforme o parâmetro angle_deg.
    @param overlay Frame onde a linha será desenhada (modificado in-place).
    @param bbox_xy Bounding box em formato [x1, y1, x2, y2].
    @param angle_deg Ângulo da orientação em graus (0 = horizontal).
    @param color Cor da linha em formato BGR (padrão: verde escuro).
    @param thickness Espessura da linha em pixels (padrão: 2).
    @return Sem retorno, modifica overlay in-place.
    """
    if angle_deg is None or math.isnan(angle_deg):
        return
    x1,y1,x2,y2 = [int(v) for v in bbox_xy]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    L = int(max(w,h) * 0.7)  # meio-comprimento da linha
    ang_rad = math.radians(angle_deg)
    dx = math.cos(ang_rad) * L
    dy = math.sin(ang_rad) * L
    xA = int(cx - dx); yA = int(cy - dy)
    xB = int(cx + dx); yB = int(cy + dy)
    cv2.line(overlay, (xA,yA), (xB,yB), color, thickness, cv2.LINE_AA)

# -----------------------
# Funções de Configuração
# -----------------------

def parse_args():
    """
    @brief Realiza o parsing dos argumentos de linha de comando.
    @details Define todos os parâmetros configuráveis para a inferência:
             modelo YOLOv8, vídeo de entrada, arquivo de saída, device (CPU/GPU),
             aliases de classes, e opções de debug.
    @return Objeto argparse.Namespace contendo todos os argumentos parseados.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Caminho do modelo YOLOv8 detect (.pt)")
    ap.add_argument("--source", required=True, help="Vídeo de entrada")
    ap.add_argument("--output", default="bbox_annot.mp4", help="Vídeo anotado de saída")
    ap.add_argument("--csv", default="bbox_metrics.csv", help="CSV de métricas por frame")
    ap.add_argument("--device", default=None, help="cpu | cuda:0 | dml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--show_all", action="store_true", help="Desenha TODAS as caixas detectadas (debug)")
    ap.add_argument("--cornea_alias", default="cornea", help="aliases p/ cornea")
    ap.add_argument("--pupil_alias", default="pupil,pupila", help="aliases p/ pupil")
    ap.add_argument("--catarata_alias", default="catarata,cataract", help="aliases p/ catarata")
    ap.add_argument("--debug_dir", default=None, help="Salvar primeiros N frames anotados")
    return ap.parse_args()

def main():
    """
    @brief Função principal que coordena toda a pipeline de detecção e anotação de vídeos.
    @details
    - Carrega o modelo YOLOv8
    - Processa cada frame do vídeo de entrada
    - Para cada frame: detecta córnea, pupila e catarata
    - Calcula métricas (áreas, confiança, distância entre centros, ângulo de incisão)
    - Desenha overlays com as detecções
    - Salva vídeo anotado e arquivo CSV com métricas
    
    @return Não retorna valor, exibe apenas mensagens de progresso
    @note A função é chamada apenas se o script for executado diretamente (__main__)
    """
    args = parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Erro abrindo vídeo {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (W, H))

    model = YOLO(args.model)

    # passar rapido para obter nomes das classes
    tmp = model.predict(np.zeros((H, W, 3), dtype=np.uint8), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
    names = tmp.names
    print("[INFO] classes do modelo:", names)

    cornea_id = match_class_id(names, args.cornea_alias)
    pupil_id = match_class_id(names, args.pupil_alias)
    catarata_id = match_class_id(names, args.catarata_alias)
    print(f"[INFO] cornea_id={cornea_id}, pupil_id={pupil_id}, catarata_id={catarata_id}")
    print("\n[INFO] Rodando... (pode demorar um pouco, dependendo do vídeo e do modelo)\n")

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    csv_fields = [
        "frame", "time_s",
        "cornea_area", "cornea_conf",
        "pupil_area", "pupil_conf",
        "catarata_area", "catarata_conf", "catarata_presence",
        "center_distance_px", "incision_angle_deg"
    ]
    csvf = open(args.csv, "w", newline="", encoding="utf-8")
    writer_csv = csv.DictWriter(csvf, fieldnames=csv_fields)
    writer_csv.writeheader()

    ema_dist = None
    fidx = 0

    # filtro de confianca minima para desenho de debug
    min_draw_conf_default = 0.25

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]

        overlay = frame.copy()

        # --- mostrar todas as caixas (debug) filtradas por confianca ---
        if args.show_all and res.boxes is not None and res.boxes.xyxy is not None:
            xyxy_all = res.boxes.xyxy.cpu().numpy()
            cls_all = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else None
            conf_all = res.boxes.conf.cpu().numpy() if getattr(res.boxes, "conf", None) is not None else None
            min_draw_conf = min_draw_conf_default
            for i, (x1, y1, x2, y2) in enumerate(xyxy_all):
                c = float(conf_all[i]) if conf_all is not None else 1.0
                if c < min_draw_conf:
                    continue
                color = (200, 200, 200)
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                if cls_all is not None:
                    name = names.get(int(cls_all[i]), str(int(cls_all[i])))
                    cv2.putText(overlay, f"{name} {c:.2f}", (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # --- funcao auxiliar: escolher a maior bbox para um ID de classe e retornar (xyxy, conf) ---
        def largest_bbox_for(target_id: Optional[int]):
            if target_id is None or res.boxes is None or res.boxes.xyxy is None or res.boxes.cls is None:
                return None
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy() if getattr(res.boxes, "conf", None) is not None else np.ones((xyxy.shape[0],))
            sel = xyxy[cls == target_id]
            sel_conf = confs[cls == target_id]
            if sel.shape[0] == 0:
                return None
            areas = (sel[:, 2] - sel[:, 0]) * (sel[:, 3] - sel[:, 1])
            i = int(np.argmax(areas))
            return (sel[i], float(sel_conf[i]))

        cornea_item = largest_bbox_for(cornea_id)  # (xyxy, conf) ou None
        pupil_item = largest_bbox_for(pupil_id)
        catarata_item = largest_bbox_for(catarata_id)

        # desenhar caixas principais, calcular centros/areas
        centers = {}
        cornea_area = float('nan'); cornea_conf = float('nan')
        pupil_area = float('nan'); pupil_conf = float('nan')
        catarata_area = float('nan'); catarata_conf = float('nan'); catarata_presence = 0
        incision_angle_deg = float('nan')

        # cores (BGR)
        cornea_color = (0, 0, 255)    # vermelho
        pupil_color = (255, 0, 0)     # azul
        catarata_color = (0, 200, 0)  # verde
        hough_color = (0, 255, 255)   # amarelo para fallback

        if cornea_item is not None:
            cor_xy, cor_conf = cornea_item
            x1, y1, x2, y2 = cor_xy.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), cornea_color, 2)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            centers["cornea"] = (cx, cy)
            cv2.circle(overlay, (int(cx), int(cy)), 3, cornea_color, -1)
            cornea_area = float((x2 - x1) * (y2 - y1))
            cornea_conf = float(cor_conf)

        if pupil_item is not None:
            pup_xy, pup_conf = pupil_item
            x1, y1, x2, y2 = pup_xy.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), pupil_color, 2)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            centers["pupil"] = (cx, cy)
            cv2.circle(overlay, (int(cx), int(cy)), 3, pupil_color, -1)
            pupil_area = float((x2 - x1) * (y2 - y1))
            pupil_conf = float(pup_conf)

        if catarata_item is not None:
            cat_xy, cat_conf = catarata_item
            x1, y1, x2, y2 = cat_xy.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), catarata_color, 2)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cv2.circle(overlay, (int(cx), int(cy)), 3, catarata_color, -1)
            catarata_area = float((x2 - x1) * (y2 - y1))
            catarata_conf = float(cat_conf)
            catarata_presence = 1
            # calcular angulo de incisao dentro desta bbox usando metodo de Hough Lines
            try:
                # extrair ROI com seguranca
                H_img, W_img = frame.shape[:2]
                xa = max(0, x1); ya = max(0, y1); xb = min(W_img-1, x2); yb = min(H_img-1, y2)
                roi = frame[ya:yb+1, xa:xb+1]
                ang = compute_dominant_orientation_deg(roi)
                if ang is not None:
                    incision_angle_deg = ang
                    # desenhar linha de orientacao dominante na sobreposicao (em verde mais escuro)
                    draw_dominant_line_on_overlay(overlay, (x1,y1,x2,y2), incision_angle_deg, color=(0,150,0), thickness=2)
            except Exception:
                incision_angle_deg = float('nan')

        # fallback Hough para pupila (se nao detectada)
        pupil_circle = None
        if "pupil" not in centers and cornea_item is not None:
            cor_xy, _ = cornea_item
            # passar bbox para Hough
            pupil_circle = find_pupil_hough_in_roi(frame, cor_xy)
            if pupil_circle is not None:
                cx, cy, rr = pupil_circle
                centers["pupil"] = (cx, cy)
                # desenhar fallback em amarelo
                cv2.circle(overlay, (int(cx), int(cy)), int(rr), hough_color, 2)
                cv2.circle(overlay, (int(cx), int(cy)), 3, hough_color, -1)

        # distancia entre centros
        center_dist = None
        if "cornea" in centers and "pupil" in centers:
            center_dist = dist(centers["cornea"], centers["pupil"])

        draw_dist = center_dist
        if args.ema > 0:
            ema_dist = ema_update(ema_dist, center_dist, args.ema)
            draw_dist = ema_dist

        # ---------------------------
        # legenda (canto superior esquerdo)
        # ---------------------------
        legend_pad_x = 10
        legend_pad_y = 10
        legend_w = 360
        legend_h = 140
        lx1, ly1 = legend_pad_x, legend_pad_y
        lx2, ly2 = lx1 + legend_w, ly1 + legend_h

        alpha = 0.6
        bg = overlay.copy()
        cv2.rectangle(bg, (lx1, ly1), (lx2, ly2), (10, 10, 10), -1)
        cv2.addWeighted(bg, alpha, overlay, 1 - alpha, 0, overlay)

        items = [
            (cornea_color, "Cornea (model)"),
            (pupil_color, "Pupil (model)"),
            (hough_color, "Pupil (Hough fallback)"),
            (catarata_color, "Catarata (model)")
        ]

        item_x = lx1 + 12
        item_y = ly1 + 26
        line_h = 28
        for color, text in items:
            cv2.rectangle(overlay, (item_x, item_y - 12), (item_x + 18, item_y + 6), color, -1)
            pos = (item_x + 26, item_y)
            cv2.putText(overlay, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            item_y += line_h

        # origem da pupila
        origin_text = "Origem da pupila: "
        if "pupil" in centers and pupil_item is not None:
            origin_text += "modelo"
        elif "pupil" in centers and pupil_item is None and pupil_circle is not None:
            origin_text += "hough (fallback)"
        else:
            origin_text += "nenhuma"
        origin_pos = (lx1 + 12, ly2 + 30)
        cv2.putText(overlay, origin_text, origin_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, origin_text, origin_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # desenhar distancia no canto superior direito (com contorno)
        if draw_dist is not None:
            dist_text = f"Dist. centro: {draw_dist:.1f}px"
            (text_w, _), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            dist_pos = (W - text_w - 10, 34)
            cv2.putText(overlay, dist_text, dist_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, dist_text, dist_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # mostrar valor do angulo de incisao na legenda (se presente)
        angle_text = f"Angulo de incisao: {incision_angle_deg:.1f}" if not math.isnan(incision_angle_deg) else "Angulo de incisao: N/A"
        angle_pos = (lx1 + 12, ly2 - 10)
        cv2.putText(overlay, angle_text, angle_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, angle_text, angle_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        # mesclar/escrever
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        writer.write(frame)

        # linha do CSV
        time_s = fidx / fps if fps and fps > 0 else 0.0
        row = {
            "frame": fidx, "time_s": time_s,
            "cornea_area": cornea_area, "cornea_conf": cornea_conf,
            "pupil_area": pupil_area, "pupil_conf": pupil_conf,
            "catarata_area": catarata_area, "catarata_conf": catarata_conf, "catarata_presence": catarata_presence,
            "center_distance_px": draw_dist if draw_dist is not None else float('nan'),
            "incision_angle_deg": incision_angle_deg if not math.isnan(incision_angle_deg) else float('nan')
        }
        writer_csv.writerow(row)

        # frames de debug
        if args.debug_dir and fidx < 50:
            cv2.imwrite(os.path.join(args.debug_dir, f"debug_{fidx:05d}.png"), frame)

        fidx += 1

    cap.release()
    writer.release()
    csvf.close()
    print(f"[OK] Vídeo anotado: {args.output}")
    print(f"[OK] Métricas CSV: {args.csv}")

if __name__ == "__main__":
    main()
