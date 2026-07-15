#!/usr/bin/env bash
# Compila libedgetpu.1.dylib para el Coral USB Accelerator en macOS arm64 (Apple Silicon).
# Reproduce las Fases 0 y 1 (ver BUILD_NOTES.md). NO usa Bazel para libedgetpu.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Versión de TensorFlow acoplada (regla de oro). libedgetpu de feranick @ TF 2.19.1.
TF_COMMIT="f4247ebb6f9e7421f38c3f01a9a5d5cd54bd24fd"     # TF 2.19.1
FB_VERSION="v24.3.25"                                     # exigido por schema_generated.h de TF 2.19.1
BREW_PREFIX="$(brew --prefix)"

echo "==> [0/5] Prerequisitos (Homebrew)"
command -v brew >/dev/null || { echo "Falta Homebrew"; exit 1; }
[ "$(uname -m)" = "arm64" ] || { echo "Este script es para arm64 (Apple Silicon)"; exit 1; }
brew list --versions libusb      >/dev/null 2>&1 || brew install libusb
brew list --versions abseil      >/dev/null 2>&1 || brew install abseil
brew list --versions flatbuffers >/dev/null 2>&1 || brew install flatbuffers  # (para flatc del sistema; usamos el 24 propio)
brew list --versions cmake       >/dev/null 2>&1 || brew install cmake

echo "==> [1/5] Clonar TensorFlow $TF_COMMIT (shallow)"
if [ ! -d tensorflow/.git ]; then
  mkdir -p tensorflow && ( cd tensorflow && git init -q && \
    git remote add origin https://github.com/tensorflow/tensorflow 2>/dev/null || true; \
    git fetch --depth 1 origin "$TF_COMMIT" && git checkout -q FETCH_HEAD )
fi

echo "==> [2/5] Clonar feranick/libedgetpu"
[ -d libedgetpu/.git ] || git clone --depth 1 https://github.com/feranick/libedgetpu

echo "==> [3/5] Compilar flatbuffers $FB_VERSION (estática) en deps/fb24"
if [ ! -f deps/fb24/lib/libflatbuffers.a ]; then
  [ -d flatbuffers/.git ] || git clone --depth 1 -b "$FB_VERSION" https://github.com/google/flatbuffers
  ( cd flatbuffers && rm -rf build && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_TESTS=OFF \
      -DFLATBUFFERS_BUILD_SHAREDLIB=OFF -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_INSTALL_PREFIX="$REPO_DIR/deps/fb24" && \
    make -j"$(sysctl -n hw.ncpu)" && make install )
fi

echo "==> [4/5] Instalar Makefile.macos y compilar libedgetpu (throttled)"
cp patches/Makefile.macos libedgetpu/makefile_build/Makefile.macos
( cd libedgetpu && \
  BREW="$BREW_PREFIX" FB="$REPO_DIR/deps/fb24" \
  TFROOT="$REPO_DIR/tensorflow/" \
  make -f makefile_build/Makefile.macos -j"$(sysctl -n hw.ncpu)" libedgetpu-throttled )

echo "==> [5/5] Copiar artefacto a dist/"
mkdir -p dist
cp -f libedgetpu/out/throttled/darwin_arm64/libedgetpu.1.0.dylib dist/
ln -sf libedgetpu.1.0.dylib dist/libedgetpu.1.dylib
ln -sf libedgetpu.1.0.dylib dist/libedgetpu.dylib

echo
echo "OK -> dist/libedgetpu.1.dylib"
file dist/libedgetpu.1.0.dylib
echo "Instalación opcional en el sistema:"
echo "  sudo cp dist/libedgetpu.1.0.dylib /usr/local/lib/ && \\"
echo "  sudo ln -sf libedgetpu.1.0.dylib /usr/local/lib/libedgetpu.1.dylib"
