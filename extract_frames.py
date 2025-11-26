#!/usr/bin/env python3
"""
@file extract_frames.py
@brief Extrai frames de um vídeo a intervalos regulares para processamento posterior.
@details Este script lê um vídeo e salva um frame a cada N quadros especificado.
         Útil para preparar datasets de treino ou análise frame-a-frame.

Exemplo de uso:
  python extract_frames.py --video data/raw/video1.mp4 --out data/frames/video1 --step 30
"""
import argparse
import os
import cv2


def main():
    """
    @brief Função principal que coordena a extração de frames do vídeo.
    @details
    - Abre o vídeo de entrada usando OpenCV
    - Itera sobre todos os frames
    - Salva 1 frame a cada N quadros (conforme parametrizado)
    - Exibe estatísticas finais de extração
    
    @return Não retorna valor, apenas exibe mensagens de progresso
    @throws RuntimeError se o vídeo não puder ser aberto
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Caminho do vídeo de entrada")
    ap.add_argument("--out", required=True, help="Pasta de saída para salvar os frames")
    ap.add_argument("--step", type=int, default=15, help="Salva 1 frame a cada N quadros (padrão=15)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {args.video}")
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.step == 0:
            out_path = os.path.join(args.out, f"frame_{idx:06d}.png")
            cv2.imwrite(out_path, frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"[OK] {saved} frames salvos em {args.out} (step={args.step})")

if __name__ == "__main__":
    main()
