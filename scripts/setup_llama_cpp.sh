#!/usr/bin/env bash
# Fetch and build llama.cpp at the commit this repository was validated against.
#
#   scripts/setup_llama_cpp.sh [install-dir]
#
# Environment overrides:
#   MODEL              boole-1.7b (default): upstream llama.cpp, into third_party/llama.cpp
#                      boole-32b: our llama.cpp branch for Boole-32B-A2.5B-preview (CUDA only),
#                                 into third_party/llama.cpp-boole-32b
#   LLAMA_CPP_COMMIT   commit to build          (default: the pinned one for MODEL)
#   CUDA_ARCH          CMake CUDA architecture  (default: native GPU, else 120)
#   GGML_CUDA=0        build the CPU backend only (Boole-1.7B only)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "${MODEL:-boole-1.7b}" in
  boole-1.7b)
    REPO="https://github.com/ggml-org/llama.cpp.git"
    PINNED_COMMIT="b42b7e6d3059503dd4bed6ca5dcf342bb6379f67"
    DEFAULT_DEST="$ROOT/third_party/llama.cpp" ;;
  boole-32b)
    # PrecisionNumerics/llama.cpp, branch mul-mat-id-multi: mixed 1/2/4-bit MoE expert kernels
    REPO="https://github.com/PrecisionNumerics/llama.cpp.git"
    PINNED_COMMIT="f88aa48cfcc5f44074db61b390a78766fbea2476"
    DEFAULT_DEST="$ROOT/third_party/llama.cpp-boole-32b"
    if [ "${GGML_CUDA:-1}" = "0" ]; then
      echo "Boole-32B-A2.5B-preview needs a CUDA GPU; GGML_CUDA=0 is not supported." >&2; exit 1
    fi ;;
  *) echo "unknown MODEL='${MODEL}' (use boole-1.7b or boole-32b)" >&2; exit 1 ;;
esac
COMMIT="${LLAMA_CPP_COMMIT:-$PINNED_COMMIT}"
DEST="${1:-$DEFAULT_DEST}"

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
cmake -S "$DEST" -B "$DEST/build" "${generator[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF "${cuda_flags[@]}"
cmake --build "$DEST/build" --config Release -j "$(nproc)" \
  --target llama-cli llama-server
echo "built: $DEST/build/bin"
