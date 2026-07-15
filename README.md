# Coral USB Accelerator en macOS arm64 (Apple Silicon)

Pila **nativa arm64** (sin Rosetta) para usar el **Google Coral USB Accelerator
(Edge TPU)** con **Python 3.12** en Macs Apple Silicon (probado en M3, macOS 26).

Google abandonó esta pila (wheels oficiales solo hasta Python 3.9, sin binarios
macOS arm64). Esto compila el runtime desde los forks de
[feranick](https://github.com/feranick) y ejecuta inferencia en el TPU real.

## Estado

| Componente | Estado |
|---|---|
| `libedgetpu.1.dylib` (runtime/driver, arm64) | ✅ compilado desde fuente |
| Intérprete TFLite (`ai-edge-litert` cp312) | ✅ |
| Inferencia en Edge TPU real | ✅ **verificada** |
| `pycoral` | ❌ no incluido — ver [BUILD_NOTES.md](BUILD_NOTES.md) |

> **pycoral** no se compila con TF 2.19 (migración *pywrap* de TensorFlow) y su
> build oficial requiere TF 2.18.1. Este proyecto usa el **Plan B**:
> `libedgetpu` + `ai-edge-litert` + `load_delegate`, que es la API real bajo
> pycoral y cubre clasificación, detección, etc.

## Prueba de éxito verificada

`parrot.jpg` → **Ara macao (Scarlet Macaw)** 0.76, con la firma inequívoca de
hardware: 1ª inferencia ~108 ms (carga del modelo al TPU), resto ~13.5 ms.
El Coral se re-enumera de `1a6e:089a` (Global Unichip, DFU) a `18d1:9302`
(Google Inc.) al subir el firmware en la primera inferencia.

## Cómo correr — paso a paso (binario precompilado)

El binario **no** viene en el repo; se descarga del
[Release](https://github.com/felipero25/coral_macos_arm64/releases/latest).
`run_edgetpu.py` busca la biblioteca en **`dist/libedgetpu.1.dylib`** (junto al
script); también puedes indicar otra ruta con `--delegate <ruta>`.

**1. Clonar el repo**
```bash
git clone https://github.com/felipero25/coral_macos_arm64.git
cd coral_macos_arm64
```

**2. Descargar la biblioteca del Release y colocarla en `dist/`**
```bash
mkdir -p dist
curl -L -o dist/libedgetpu.1.0.dylib \
  https://github.com/felipero25/coral_macos_arm64/releases/download/v0.1.0/libedgetpu.1.0.dylib
ln -sf libedgetpu.1.0.dylib dist/libedgetpu.1.dylib      # nombre que espera el script
```
> Alternativa: descarga el `.tar.gz` del Release (ya trae los symlinks) y
> descomprímelo dentro de `dist/`.

**3. Instalar dependencias del sistema (Homebrew)**
```bash
brew install libusb abseil
```
El dylib enlaza `libusb` y `abseil` de Homebrew por **ruta absoluta**; son
obligatorias (ver *Notas*).

**4. Entorno Python 3.12 + dependencias**
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**5. Conectar el Coral USB Accelerator y ejecutar**
```bash
python run_edgetpu.py \
  --model testdata/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
  --labels testdata/inat_bird_labels.txt \
  --input testdata/parrot.jpg
```

**6. Señal de éxito**: la 1ª inferencia tarda ~100 ms (carga del modelo al TPU) y
las siguientes ~13 ms; clasifica `parrot.jpg` como *Ara macao (Scarlet Macaw)*.
Si todas las inferencias tardan igual, está corriendo en CPU (no cuenta).

> ¿Prefieres no depender de `dist/`? Instala el dylib en el sistema:
> `sudo cp dist/libedgetpu.1.0.dylib /usr/local/lib/ && sudo ln -sf libedgetpu.1.0.dylib /usr/local/lib/libedgetpu.1.dylib`
> y pasa `--delegate /usr/local/lib/libedgetpu.1.dylib`.

## Ejemplo: detección de objetos con la cámara

`detect_camera.py` hace detección de objetos en tiempo real (SSD MobileNet v2
COCO) sobre el Edge TPU y la webcam.

```bash
# En vivo (pulsa 'q' para salir):
python detect_camera.py

# Prueba sobre una imagen (sin cámara), guarda anotada:
python detect_camera.py --input testdata/parrot.jpg --output /tmp/out.jpg
```

Requiere el modelo COCO (en `testdata/`) y `opencv-python` (en `requirements.txt`).
En **macOS**, la primera vez debes dar **permiso de cámara a tu terminal** en
*Ajustes del sistema → Privacidad y seguridad → Cámara* y reiniciar la terminal.
Muestra ms de inferencia y FPS en la ventana.

## Compilar desde fuente

```bash
./scripts/build_libedgetpu.sh
```

Reproduce las Fases 0–1: instala deps de Homebrew, clona TensorFlow 2.19.1 y
`feranick/libedgetpu`, compila **flatbuffers 24.3.25** (estático), aplica
`patches/Makefile.macos` y compila `libedgetpu` (target *throttled*, menos calor).
Salida en `dist/`.

## Notas importantes

- **Versión de TF**: `libedgetpu` se compila contra **TF 2.19.1** (pin activo de
  feranick). Ver la "regla de oro" en [BUILD_NOTES.md](BUILD_NOTES.md).
- **Dependencia de runtime**: el dylib enlaza `abseil` y `libusb` de Homebrew por
  ruta absoluta (`/opt/homebrew/...`). El binario precompilado solo funciona en
  Macs con esas fórmulas de Homebrew instaladas; si actualizas abseil de forma
  incompatible, recompila.
- **throttled vs direct**: se compila *throttled* (~13.5 ms/inferencia) para no
  calentar el dispositivo. Para ~3 ms usa el target `libedgetpu` (direct).
- **`system_profiler SPUSBDataType`** sale vacío en macOS reciente; usa
  `ioreg -p IOUSB -l -w 0` para ver el USB.

## Estructura

```
run_edgetpu.py              Clasificación en el TPU (Plan B, load_delegate)
detect_camera.py            Detección de objetos en vivo (cámara + Edge TPU)
requirements.txt            ai-edge-litert, pillow, numpy, opencv (cp312 arm64)
patches/Makefile.macos      Makefile adaptado a macOS/ld64 (build USB-only)
scripts/build_libedgetpu.sh Build reproducible de libedgetpu
dist/                       (no versionado) aquí va el .dylib del Release o del build
testdata/                   modelo edgetpu + labels + imagen
BUILD_NOTES.md              Bitácora detallada de workarounds
```
