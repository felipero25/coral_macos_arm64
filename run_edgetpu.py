#!/usr/bin/env python3
"""Plan B: inferencia en el Edge TPU cargando el delegate directamente
(sin pycoral), usando ai-edge-litert. Mide el tiempo por inferencia para
confirmar que corre en el TPU (1a lenta ~10-15ms, resto ~3ms)."""
import os, time, argparse
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter, load_delegate

# Delegate por defecto: dist/libedgetpu.1.dylib junto a este script.
_HERE = os.path.dirname(os.path.abspath(__file__))
DELEGATE = os.path.join(_HERE, "dist", "libedgetpu.1.dylib")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--delegate", default=DELEGATE)
    ap.add_argument("--count", type=int, default=6)
    args = ap.parse_args()

    print(f"[*] Cargando delegate: {args.delegate}")
    delegate = load_delegate(args.delegate)
    print("[+] Delegate cargado OK")

    interpreter = Interpreter(model_path=args.model,
                              experimental_delegates=[delegate])
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    _, h, w, _ = inp["shape"]
    print(f"[*] input {inp['shape']} {inp['dtype']} | output {out['shape']} {out['dtype']}")

    img = Image.open(args.input).convert("RGB").resize((w, h), Image.LANCZOS)
    x = np.asarray(img, dtype=inp["dtype"])[np.newaxis, ...]

    labels = {}
    with open(args.labels) as f:
        for i, line in enumerate(f):
            labels[i] = line.strip()

    print(f"\n[*] Ejecutando {args.count} inferencias:")
    times = []
    for i in range(args.count):
        interpreter.set_tensor(inp["index"], x)
        t0 = time.perf_counter()
        interpreter.invoke()
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        print(f"    inferencia {i}: {dt:7.2f} ms")

    scores = interpreter.get_tensor(out["index"])[0].astype(np.float32)
    if out["dtype"] == np.uint8:
        scale, zp = out["quantization"]
        scores = scale * (scores - zp)
    top = np.argsort(scores)[-3:][::-1]
    print("\n[*] Top-3:")
    for k in top:
        print(f"    {labels.get(int(k), k)}: {scores[k]:.4f}")

    print(f"\n[=] 1a inferencia: {times[0]:.2f} ms | media resto: "
          f"{np.mean(times[1:]):.2f} ms")
    if times[0] > 1.6 * np.mean(times[1:]):
        print("[✓] FIRMA DE EDGE TPU: 1a inferencia notablemente más lenta "
              "(carga del modelo al TPU). Corriendo en HARDWARE.")
    else:
        print("[!] Tiempos uniformes: podría estar en CPU. Revisar.")

if __name__ == "__main__":
    main()
