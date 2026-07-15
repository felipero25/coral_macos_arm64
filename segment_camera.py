#!/usr/bin/env python3
"""Segmentación semántica en tiempo real con DeepLabV3 sobre el Edge TPU.

Modelo: deeplabv3_mnv2_pascal (Pascal VOC, 21 clases). Pinta una máscara de
color por píxel y la mezcla con la imagen. Plan B (ai-edge-litert, sin pycoral).

Uso:
    python segment_camera.py                    # cámara en vivo ('q' para salir)
    python segment_camera.py --input foto.jpg --output /tmp/seg.jpg
"""
import os
import time
import argparse
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter, load_delegate

_HERE = os.path.dirname(os.path.abspath(__file__))
DELEGATE = os.path.join(_HERE, "dist", "libedgetpu.1.dylib")
MODEL = os.path.join(_HERE, "testdata", "deeplabv3_mnv2_pascal_quant_edgetpu.tflite")

PASCAL_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                  "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
                  "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                  "train", "tvmonitor"]


def voc_colormap(n=256):
    """Colormap estándar de Pascal VOC (bit-shift)."""
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


COLORMAP = voc_colormap()


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
    seg = interp.get_tensor(interp.get_output_details()[0]["index"])[0]  # [H,W] clase
    return seg.astype(np.uint8), dt


def overlay(frame_bgr, seg):
    h, w = frame_bgr.shape[:2]
    # colorear máscara (RGB->BGR) y redimensionar al frame
    color = COLORMAP[seg]  # [H,W,3] RGB
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    color = cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize((seg > 0).astype(np.uint8) * 255, (w, h),
                      interpolation=cv2.INTER_NEAREST)
    blend = cv2.addWeighted(frame_bgr, 0.5, color, 0.5, 0)
    # solo mezclar donde hay clase != background
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
    out = np.where(m3, blend, frame_bgr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--delegate", default=DELEGATE)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--input", help="imagen para modo prueba (sin cámara)")
    ap.add_argument("--output", default="/tmp/seg_out.jpg")
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
            seg, dt = infer(interp, rgb, (iw, ih))
            print(f"    inferencia {i}: {dt:6.2f} ms")
        out = overlay(frame, seg)
        cv2.imwrite(args.output, out)
        present = sorted(set(seg.ravel().tolist()))
        names = [PASCAL_CLASSES[c] for c in present if c < len(PASCAL_CLASSES)]
        print(f"[*] clases detectadas: {names}")
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
        seg, dt = infer(interp, rgb, (iw, ih))
        out = overlay(frame, seg)
        fps = 0.9 * fps + 0.1 * (1000.0 / dt) if dt > 0 else fps
        cv2.putText(out, f"TPU {dt:.1f} ms | {fps:.0f} FPS", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Coral Edge TPU - segmentacion (q para salir)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
