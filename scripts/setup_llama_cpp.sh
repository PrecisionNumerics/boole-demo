#!/usr/bin/env bash
# Fetch and build llama.cpp at the commit this repository was validated against.
#
#   scripts/setup_llama_cpp.sh [install-dir]
#
# Environment overrides:
#   LLAMA_CPP_COMMIT   commit to build          (default: the pinned one below)
#   CUDA_ARCH          CMake CUDA architecture  (default: native GPU, else 120)
#   GGML_CUDA=0        build the CPU backend only (Boole-1.7B; Boole-32B-A2.5B-Preview needs CUDA)
#   HF_UI_VERSION      llama-server web UI assets to download (default: b12359, those of llama.cpp v1.0.0)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# PrecisionNumerics/llama.cpp: llama.cpp v1.0.0 plus our mixed-precision MoE expert kernels and decode kernels for one or
# several concurrent requests (branch v1.0-flint-decode)
REPO="https://github.com/PrecisionNumerics/llama.cpp.git"
PINNED_COMMIT="c89e93d5763e493f94032bffab4aae92edd2fec0"
DEFAULT_DEST="$ROOT/third_party/llama.cpp"
COMMIT="${LLAMA_CPP_COMMIT:-$PINNED_COMMIT}"
DEST="${1:-$DEFAULT_DEST}"
# the build picks the web UI by its build number (the commit count), which a shallow fetch does not have: use the UI of
# the llama.cpp release the fork is based on
export HF_UI_VERSION="${HF_UI_VERSION:-b12359}"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$DEST"
  git -C "$DEST" init -q
  git -C "$DEST" remote add origin "$REPO"
fi
git -C "$DEST" remote set-url origin "$REPO"
git -C "$DEST" fetch --depth 1 origin "$COMMIT"
git -C "$DEST" checkout -q FETCH_HEAD
echo "llama.cpp at $(git -C "$DEST" rev-parse HEAD)"

cuda_flags=()
if [ "${GGML_CUDA:-1}" != "0" ]; then
  arch="${CUDA_ARCH:-}"
  if [ -z "$arch" ] && command -v nvidia-smi >/dev/null; then
    arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
  fi
  cuda_flags=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=${arch:-120}")
fi

generator=()
command -v ninja >/dev/null && generator=(-G Ninja)
cmake -S "$DEST" -B "$DEST/build" "${generator[@]}" -DCMAKE_BUILD_TYPE=Release "${cuda_flags[@]}"
cmake --build "$DEST/build" --config Release -j "$(nproc)" \
  --target llama-cli llama-server
echo "built: $DEST/build/bin"
