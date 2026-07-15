#!/usr/bin/env python3
"""Detección de objetos en tiempo real con el Google Coral Edge TPU + cámara.

Modelo: SSD MobileNet v2 (COCO), cuantizado, compilado para Edge TPU.
Corre el delegate directamente con ai-edge-litert (Plan B, sin pycoral).

Uso (cámara en vivo, pulsa 'q' para salir):
    python detect_camera.py

Uso (una imagen, para probar sin cámara — guarda anotada):
    python detect_camera.py --input testdata/parrot.jpg --output /tmp/out.jpg
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
                     "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
LABELS = os.path.join(_HERE, "testdata", "coco_labels.txt")


def load_labels(path):
    with open(path) as f:
        return [line.strip() for line in f]  # índice de línea == id de clase


def make_interpreter(model, delegate_path):
    print(f"[*] Cargando delegate Edge TPU: {delegate_path}")
    delegate = load_delegate(delegate_path)
    interp = Interpreter(model_path=model, experimental_delegates=[delegate])
    interp.allocate_tensors()
    return interp


def get_output_map(interp):
    """Índices de los tensores de salida (boxes, classes, scores, count).

    El modelo SSD usa la op TFLite_Detection_PostProcess, cuyos tensores se
    nombran de forma fija: `...PostProcess`=boxes, `:1`=classes, `:2`=scores,
    `:3`=count. Mapeamos por nombre (determinista); si no coincide, por forma.
    """
    details = interp.get_output_details()
    by_name = {}
    for d in details:
        name = d["name"]
        if name.endswith(":1"):
            by_name["classes"] = d["index"]
        elif name.endswith(":2"):
            by_name["scores"] = d["index"]
        elif name.endswith(":3"):
            by_name["count"] = d["index"]
        elif "PostProcess" in name:
            by_name["boxes"] = d["index"]
    if len(by_name) == 4:
        return by_name["boxes"], by_name["classes"], by_name["scores"], by_name["count"]

    # Fallback por forma: boxes=[.,N,4], count=size 1; de los [.,N] restantes,
    # classes son enteros y scores están en [0,1] (requiere valores tras invoke).
    boxes = classes = scores = count = None
    leftover = []
    for d in sorted(details, key=lambda d: d["index"]):
        shp = d["shape"]
        if len(shp) == 3 and shp[-1] == 4:
            boxes = d["index"]
        elif int(np.prod(shp)) == 1:
            count = d["index"]
        else:
            leftover.append(d["index"])
    if len(leftover) == 2:
        a, b = leftover
        va = interp.get_tensor(a).ravel()
        # scores en [0,1] y NO todos enteros; classes son enteros.
        a_is_scores = va.size and va.max() <= 1.0 and not np.all(va == np.round(va))
        scores, classes = (a, b) if a_is_scores else (b, a)
    return boxes, classes, scores, count


def detect(interp, omap, frame_rgb, size):
    boxes_i, classes_i, scores_i, count_i = omap
    inp = interp.get_input_details()[0]
    img = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)
    x = np.asarray(img, dtype=inp["dtype"])[np.newaxis, ...]
    interp.set_tensor(inp["index"], x)
    t0 = time.perf_counter()
    interp.invoke()
    dt = (time.perf_counter() - t0) * 1000
    boxes = interp.get_tensor(boxes_i)[0]
    classes = interp.get_tensor(classes_i)[0].astype(int)
    scores = interp.get_tensor(scores_i)[0]
    n = int(interp.get_tensor(count_i)[0]) if count_i is not None else len(scores)
    return boxes[:n], classes[:n], scores[:n], dt


def draw(frame_bgr, boxes, classes, scores, labels, threshold):
    h, w = frame_bgr.shape[:2]
    for box, cls, score in zip(boxes, classes, scores):
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        x0, y0 = int(xmin * w), int(ymin * h)
        x1, y1 = int(xmax * w), int(ymax * h)
        label = labels[cls] if 0 <= cls < len(labels) else str(cls)
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame_bgr, (x0, y0 - th - 6), (x0 + tw, y0), (0, 255, 0), -1)
        cv2.putText(frame_bgr, text, (x0, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame_bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--delegate", default=DELEGATE)
    ap.add_argument("--camera", type=int, default=0, help="índice de cámara")
    ap.add_argument("--input", help="imagen para modo prueba (sin cámara)")
    ap.add_argument("--output", default="/tmp/detect_out.jpg",
                    help="salida anotada en modo --input")
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()

    labels = load_labels(args.labels)
    interp = make_interpreter(args.model, args.delegate)
    _, ih, iw, _ = interp.get_input_details()[0]["shape"]
    omap = get_output_map(interp)
    print(f"[*] input {iw}x{ih} | tensores salida (boxes,classes,scores,count)={omap}")

    # ---- Modo imagen (prueba headless) ----
    if args.input:
        frame = cv2.imread(args.input)
        if frame is None:
            raise SystemExit(f"No pude leer {args.input}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # dos pasadas: la 1a incluye la carga del modelo al TPU
        for i in range(3):
            boxes, classes, scores, dt = detect(interp, omap, rgb, (iw, ih))
            print(f"    inferencia {i}: {dt:6.2f} ms")
        draw(frame, boxes, classes, scores, labels, args.threshold)
        cv2.imwrite(args.output, frame)
        dets = [(labels[c] if 0 <= c < len(labels) else c, float(s))
                for c, s in zip(classes, scores) if s >= args.threshold]
        print(f"[*] detecciones (>= {args.threshold}): {dets}")
        print(f"[=] imagen anotada -> {args.output}")
        return

    # ---- Modo cámara en vivo ----
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la cámara {args.camera}. "
                         "En macOS, da permiso de cámara a tu terminal en "
                         "Ajustes > Privacidad y seguridad > Cámara.")
    print("[*] Cámara abierta. Pulsa 'q' en la ventana para salir.")
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, classes, scores, dt = detect(interp, omap, rgb, (iw, ih))
        draw(frame, boxes, classes, scores, labels, args.threshold)
        fps = 0.9 * fps + 0.1 * (1000.0 / dt) if dt > 0 else fps
        cv2.putText(frame, f"TPU {dt:.1f} ms | {fps:.0f} FPS", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Coral Edge TPU - deteccion (q para salir)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
