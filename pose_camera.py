#!/usr/bin/env python3
"""Estimación de pose (esqueleto) en tiempo real con MoveNet sobre el Edge TPU.

Modelo: movenet_single_pose_lightning (una persona, 17 keypoints COCO).
Corre el delegate directamente con ai-edge-litert (Plan B, sin pycoral).

Uso:
    python pose_camera.py                       # cámara en vivo ('q' para salir)
    python pose_camera.py --input foto.jpg --output /tmp/pose.jpg
"""
import os
import time
import argparse
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter, load_delegate

_HERE = os.path.dirname(os.path.abspath(__file__))
DELEGATE = os.path.join(_HERE, "dist", "libedgetpu.1.dylib")
MODEL = os.path.join(_HERE, "testdata",
                     "movenet_single_pose_lightning_ptq_edgetpu.tflite")

# 17 keypoints COCO: 0 nariz,1/2 ojos,3/4 orejas,5/6 hombros,7/8 codos,
# 9/10 muñecas,11/12 caderas,13/14 rodillas,15/16 tobillos.
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16)]


def make_interpreter(model, delegate_path):
    print(f"[*] Cargando delegate Edge TPU: {delegate_path}")
    interp = Interpreter(model_path=model,
                         experimental_delegates=[load_delegate(delegate_path)])
    interp.allocate_tensors()
    return interp


def infer(interp, frame_rgb, size):
    inp = interp.get_input_details()[0]
    img = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)
    x = np.asarray(img, dtype=inp["dtype"])[np.newaxis, ...]
    interp.set_tensor(inp["index"], x)
    t0 = time.perf_counter()
    interp.invoke()
    dt = (time.perf_counter() - t0) * 1000
    # salida [1,1,17,3] -> (y, x, score) normalizados
    kpts = interp.get_tensor(interp.get_output_details()[0]["index"])[0, 0]
    return kpts, dt


def draw(frame_bgr, kpts, threshold):
    h, w = frame_bgr.shape[:2]
    pts = [(int(x * w), int(y * h), s) for y, x, s in kpts]
    for (a, b) in EDGES:
        if pts[a][2] >= threshold and pts[b][2] >= threshold:
            cv2.line(frame_bgr, pts[a][:2], pts[b][:2], (0, 255, 255), 2)
    for (px, py, s) in pts:
        if s >= threshold:
            cv2.circle(frame_bgr, (px, py), 4, (0, 0, 255), -1)
    return frame_bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--delegate", default=DELEGATE)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--input", help="imagen para modo prueba (sin cámara)")
    ap.add_argument("--output", default="/tmp/pose_out.jpg")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    interp = make_interpreter(args.model, args.delegate)
    _, ih, iw, _ = interp.get_input_details()[0]["shape"]
    print(f"[*] input {iw}x{ih}")

    if args.input:
        frame = cv2.imread(args.input)
        if frame is None:
            raise SystemExit(f"No pude leer {args.input}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for i in range(3):
            kpts, dt = infer(interp, rgb, (iw, ih))
            print(f"    inferencia {i}: {dt:6.2f} ms")
        draw(frame, kpts, args.threshold)
        cv2.imwrite(args.output, frame)
        vis = int((kpts[:, 2] >= args.threshold).sum())
        print(f"[*] keypoints visibles (>= {args.threshold}): {vis}/17")
        print(f"[=] imagen anotada -> {args.output}")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la cámara {args.camera}. En macOS da "
                         "permiso de cámara a la terminal en Ajustes > Privacidad.")
    print("[*] Cámara abierta. Pulsa 'q' para salir.")
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kpts, dt = infer(interp, rgb, (iw, ih))
        draw(frame, kpts, args.threshold)
        fps = 0.9 * fps + 0.1 * (1000.0 / dt) if dt > 0 else fps
        cv2.putText(frame, f"TPU {dt:.1f} ms | {fps:.0f} FPS", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Coral Edge TPU - pose (q para salir)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
