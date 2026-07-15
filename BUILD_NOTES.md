# BUILD_NOTES.md — Coral USB Accelerator en macOS arm64 (M3)

Bitácora de workarounds aplicados, para reproducibilidad. Ver `README.md` para el resumen.

## Versión de TF fijada para toda la pila

**TensorFlow 2.19.1** — commit `f4247ebb6f9e7421f38c3f01a9a5d5cd54bd24fd`
(SHA256 `b4cbd0371…`), que es el pin **activo** en `feranick/libedgetpu`
(`workspace.bzl`).

⚠️ Discrepancia detectada entre repos de feranick:
- `libedgetpu/workspace.bzl` (activo) → TF **2.19.1** (`f4247ebb…`)
- `pycoral/WORKSPACE` → TF **2.18.1** (`cb64295e…`)

Decisión (usuario): fijar todo a **2.19.1**. En la Fase 3 habrá que **subir el
WORKSPACE de pycoral de 2.18.1 → 2.19.1** para respetar la regla de oro.

## Fase 0 — Prerequisitos (COMPLETADA)

- arch: `arm64` ✓ · Python `3.12.11` arm64 ✓ · Xcode.app ✓ · Homebrew 5.1.10 ✓
- `brew install`: `libusb` 1.0.29, `flatbuffers` 25.12.19, `bazelisk` 1.29.0,
  `abseil` 20260107.1
- Apple clang 21.0.0 (no hay `gcc`/`gold` reales; `gcc`→clang).

## Fase 1 — libedgetpu.1.dylib arm64 (vía Makefile)

TF clonado en `coral/tensorflow` (shallow, commit exacto de 2.19.1).

El `makefile_build/Makefile` de feranick es **100% Linux**. Creado
`makefile_build/Makefile.macos` con los siguientes cambios:

### 1. Compilador y flags
- `CC=clang`, `CXX=clang++`, `FLATC=$(BREW)/bin/flatc`.
- Añadido `-arch arm64`, `-I$(BREW)/include` (Homebrew), `-Wno-deprecated-declarations`.
- `BREW ?= /opt/homebrew`.

### 2. Flags de linker (GNU ld/gold → ld64 de Apple)
Sustituidos/eliminados (ld64 no los soporta):
- `-shared` → `-dynamiclib`
- `-Wl,--soname,libedgetpu.so.1` → `-install_name @rpath/libedgetpu.1.dylib`
- `-Wl,--version-script=…lds` → **eliminado** (ld64 usa otro formato; con
  visibilidad por defecto los símbolos `edgetpu_*`/`tflite_plugin_*` quedan exportados)
- `-Wl,-Map=…` → eliminado
- `-fuse-ld=gold` → eliminado (no hay gold en macOS)
- `-L$(BREW)/lib` añadido.
- **abseil**: los `-labsl_*` explícitos del original no bastan (grafo de
  dependencias denso; además `libabsl_flags.dylib` ya no existe en abseil 20260107).
  Solución: `ABSL_LIBS := $(shell ls $(BREW)/lib/libabsl_*.dylib | sed …)` enlaza
  **todas** las libabsl_* instaladas (93). En macOS el orden de dylibs es irrelevante.

### 2b. flatbuffers 24.3.25 (choque de versión con Homebrew v25)
`schema_generated.h` de TF 2.19.1 tiene `static_assert(FLATBUFFERS_VERSION == 24.3.25)`.
Homebrew instala v25 → error de compilación. Construido flatbuffers **v24.3.25**
desde fuente como **estática** e instalada en `deps/fb24`:
```bash
git clone --depth 1 -b v24.3.25 https://github.com/google/flatbuffers
cmake .. -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_TESTS=OFF \
  -DFLATBUFFERS_BUILD_SHAREDLIB=OFF -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_INSTALL_PREFIX=<coral>/deps/fb24
make -j8 && make install
```
En el Makefile: var `FB ?= <coral>/deps/fb24`; `FLATC=$(FB)/bin/flatc`;
`-I$(FB)/include` **antes** de `-I$(BREW)/include`; se enlaza `$(FB)/lib/libflatbuffers.a`
(estática → sin dependencia en runtime).

### 3. Fuentes eliminadas del build (path PCI/gasket kernel = Linux-only; el USB
Accelerator usa el path USB vía libusb, que se auto-registra con
`REGISTER_DRIVER_PROVIDER(BeagleUsbDriverProvider)` y no depende de ellas):
- `driver/beagle/beagle_pci_driver_provider_linux.cc`  (auto-registra PCI + factory funcs Linux)
- `driver/beagle/beagle_pci_driver_provider.cc`        (base PCI abstracta, usa clases kernel gasket)
- `driver/beagle/beagle_kernel_top_level_handler.cc`   (→ beagle_ioctl.h → gasket_ioctl.h → `<linux/ioctl.h>`)
- `driver/kernel/linux/kernel_coherent_allocator_linux.cc`
- `driver/kernel/linux/kernel_event_handler_linux.cc`  (`<sys/eventfd.h>`, no existe en macOS)
- `driver/kernel/linux/kernel_event_linux.cc`
- `driver/kernel/linux/kernel_registers_linux.cc`
- `driver/kernel/kernel_coherent_allocator.cc`         (→ gasket_ioctl.h → `<linux/ioctl.h>`)
- `driver/kernel/kernel_event_handler.cc`
- `driver/kernel/kernel_interrupt_handler.cc`
- `driver/kernel/kernel_mmu_mapper.cc`                 (→ gasket_ioctl.h)
- `driver/kernel/kernel_registers.cc`

El USB provider usa `BeagleTopLevelHandler` (no el `*_kernel_*`).

### 4. Salida
- `out/{throttled,direct}/darwin_arm64/libedgetpu.1.0.dylib` + symlink `libedgetpu.1.dylib`.
- Se compila el target **throttled** (menos calor en el USB Accelerator).

### Comando de build
```bash
cd libedgetpu
TFROOT=<ruta>/coral/tensorflow/ make -f makefile_build/Makefile.macos -j8 libedgetpu-throttled
```

### Resultado (✅ ÉXITO)
- `out/throttled/darwin_arm64/libedgetpu.1.0.dylib` — Mach-O **arm64**, 4.0 MB.
- Símbolos exportados verificados: `edgetpu_create_delegate`, `edgetpu_free_delegate`,
  `edgetpu_version`, `tflite_plugin_create_delegate`, `tflite_plugin_destroy_delegate`.
- `dlopen()` OK (por ruta y por nombre con `DYLD_LIBRARY_PATH`).
- `otool -L`: depende de `libusb-1.0` (Homebrew), `libc++`/`libSystem` (sistema) y
  93 `libabsl_*` (rutas absolutas Homebrew). flatbuffers va estático (no aparece).
- Copiado a `coral/dist/` con symlinks `libedgetpu.1.dylib` y `libedgetpu.dylib`.

### Instalación en /usr/local/lib (requiere sudo — pendiente de ejecutar por el usuario)
```bash
sudo cp dist/libedgetpu.1.0.dylib /usr/local/lib/
sudo ln -sf libedgetpu.1.0.dylib /usr/local/lib/libedgetpu.1.dylib
sudo ln -sf libedgetpu.1.0.dylib /usr/local/lib/libedgetpu.dylib
```
Alternativa sin sudo: usar `dist/` directamente vía `DYLD_LIBRARY_PATH=<coral>/dist`
o pasando la ruta absoluta a `load_delegate(...)`.

## Fase 2 — Intérprete TFLite cp312 arm64 (COMPLETADA)
- venv en `coral/.venv` (Python 3.12.11 arm64).
- `pip install ai-edge-litert` → **2.1.6** (opción A; wheel arm64 cp312 nativo).
  Expone `from ai_edge_litert.interpreter import Interpreter, load_delegate`.
- También `pillow` 12.3.0 y `numpy` 2.5.1 para preprocesado.

## Fase 4 — Verificación en Edge TPU real (✅ ÉXITO, vía Plan B)
Script `run_edgetpu.py` (Plan B: `load_delegate(dist/libedgetpu.1.dylib)`, sin pycoral).
Datos de prueba en `testdata/` (descargados de `google-coral/test_data`).

```bash
source .venv/bin/activate
python run_edgetpu.py \
  --model testdata/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
  --labels testdata/inat_bird_labels.txt --input testdata/parrot.jpg
```

Resultado:
- Clasificación correcta: **Ara macao (Scarlet Macaw) 0.76**.
- 1a inferencia **108 ms**, resto **~13.5 ms** → firma inequívoca de Edge TPU.
- Re-enumeración USB confirmada por `ioreg`: `1a6e:089a` (Global Unichip, DFU) →
  `18d1:9302` (Google Inc.) tras subir el firmware en la 1a inferencia.

Nota sobre velocidad: ~13.5 ms/inferencia (no ~3 ms) porque se compiló el target
**throttled** (frecuencia reducida, para no calentar el
USB Accelerator). Para ~3 ms compilar el target `libedgetpu` (direct) en su lugar.

`system_profiler SPUSBDataType` devuelve vacío en esta macOS (Darwin 27); usar
`ioreg -p IOUSB -l -w 0` para inspeccionar USB. IDs en decimal: Unichip 6766/2202,
Google 6353/37634.

## Fase 3 — pycoral wheel cp312 macosx_arm64 (DESCARTADO)

**Resultado: no viable con TF 2.19; se optó por el Plan B.** En TF 2.19 TensorFlow
completó la migración *pywrap*: el macro `pybind_extension` ya no emite el `.so`
directamente (`//src:_pywrap_coral` genera solo un *info wrapper*; el `.so` real lo
enlaza un target agregador `pywrap_library` que el `src/BUILD` de pycoral no tiene).
El build oficial de pycoral requiere **TF 2.18.1**. Como libedgetpu ya está en
2.19.1 y funciona, y el objetivo se cumple con Plan B, no se persiguió pycoral.
El intento contra 2.18.1 (skew controlado) quedó a medias y se canceló.

Lo de abajo documenta el setup que se llegó a hacer, por si se retoma:

### Setup realizado antes de descartar

pycoral SÍ trae soporte `darwin_arm64` en su Makefile (usa Bazel). Setup:

1. **WORKSPACE bump a TF 2.19.1**: cambiado el `http_archive` de `org_tensorflow`
   de `cb64295e` (2.18.1) → `f4247ebb` (2.19.1) + su sha256 y strip_prefix.
2. **Submódulos**: `git submodule update --init --recursive --depth 1` para
   `libcoral`, `test_data`; y aparte `libedgetpu` (nivel pycoral). El nested
   `libcoral/libedgetpu` ya viene en tag `16.0TF2.19.1-1`.
3. **Bump del submódulo `pycoral/libedgetpu/workspace.bzl`** a TF 2.19.1 también
   (el WORKSPACE llama `libedgetpu_dependencies()` desde ahí).
4. **libedgetpu precompilado**: copiado nuestro dylib a
   `pycoral/libedgetpu_bin/direct/darwin_arm64/libedgetpu.1.0.dylib` (+ symlinks),
   que es donde el Makefile lo enlaza (`-L… -ledgetpu.1`).
5. **Bazel 6.5.0**: `.bazelversion` = `6.5.0`; `bazel`→bazelisk. Env de build:
   `USE_BAZEL_VERSION=6.5.0 HERMETIC_PYTHON_VERSION=3.12 TF_PYTHON_VERSION=3.12`.

### Fix: `USE_PYWRAP_RULES` no definido (WORKSPACE 2.18.1 vs TF 2.19)
Al bumpear a TF 2.19, `pywrap.default.bzl` hace
`load("@python_version_repo//:py_version.bzl", "USE_PYWRAP_RULES")`, pero la regla
legacy `python_repository` de pycoral (heredada de 2.18.1) no emite ese símbolo
(y además ignoraba `HERMETIC_PYTHON_VERSION`, leía `TF_PYTHON_VERSION`, default 3.11).
**Solución (igual que feranick/libedgetpu)**: sustituir en el WORKSPACE
`python_repository(name="python_version_repo")` por un
`local_repository(path="local_repo/python_version_repo")` con `py_version.bzl`
escrito a mano (copiado de libedgetpu) que fija Python 3.12 y `USE_PYWRAP_RULES = "1"`.

Comando de build:
```bash
cd pycoral
USE_BAZEL_VERSION=6.5.0 HERMETIC_PYTHON_VERSION=3.12 TF_PYTHON_VERSION=3.12 \
  make CPU=darwin_arm64 PYTHON=<coral>/.venv/bin/python pybind
```

## Dependencia de runtime importante
El dylib enlaza abseil de Homebrew por **ruta absoluta**
(`/opt/homebrew/opt/abseil/lib/...`). No desinstalar/actualizar de forma incompatible
`abseil` (20260107.1), `libusb` (1.0.29) ni cambiar el prefijo de Homebrew, o el
delegate no cargará. Para pycoral (Fase 3) hay que respetar esta misma abseil.
