#!/usr/bin/env bash
# Fetch and build llama.cpp at the commit this repository was validated against.
#
#   scripts/setup_llama_cpp.sh [install-dir]      (default: third_party/llama.cpp)
#
# Environment overrides:
#   LLAMA_CPP_COMMIT   commit to build          (default: the pinned one below)
#   CUDA_ARCH          CMake CUDA architecture  (default: native GPU, else 120)
#   GGML_CUDA=0        build the CPU backend only
set -euo pipefail

PINNED_COMMIT="b42b7e6d3059503dd4bed6ca5dcf342bb6379f67"
COMMIT="${LLAMA_CPP_COMMIT:-$PINNED_COMMIT}"
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/third_party/llama.cpp}"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$DEST"
  git -C "$DEST" init -q
  git -C "$DEST" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
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
cmake -S "$DEST" -B "$DEST/build" "${generator[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF "${cuda_flags[@]}"
cmake --build "$DEST/build" --config Release -j "$(nproc)" \
  --target llama-cli llama-server llama-bench llama-perplexity llama-completion
echo "built: $DEST/build/bin"
