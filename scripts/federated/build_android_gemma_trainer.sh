#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <android-ndk-root> [arm64-v8a] [30]" >&2
  echo "   or: $0 [arm64-v8a] [30]    # auto-detect NDK root" >&2
}

resolve_ndk_root() {
  local candidate=""
  if [[ -n "${ANDROID_NDK_ROOT:-}" ]]; then
    candidate="$ANDROID_NDK_ROOT"
  elif [[ -n "${ANDROID_NDK_HOME:-}" ]]; then
    candidate="$ANDROID_NDK_HOME"
  elif [[ -d "${HOME}/Library/Android/sdk/ndk" ]]; then
    candidate="$(find "${HOME}/Library/Android/sdk/ndk" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  fi
  if [[ -z "$candidate" || ! -d "$candidate" ]]; then
    echo "Could not auto-detect Android NDK root. Set ANDROID_NDK_ROOT or pass the path explicitly." >&2
    exit 1
  fi
  (cd "$candidate" && pwd)
}

host_tag() {
  local os arch prebuilt_root
  os="$(uname -s)"
  arch="$(uname -m)"
  prebuilt_root="$NDK_ROOT/toolchains/llvm/prebuilt"
  case "$os" in
    Linux)
      [[ -d "$prebuilt_root/linux-x86_64" ]] && { echo "linux-x86_64"; return; }
      ;;
    Darwin)
      [[ "$arch" == "arm64" && -d "$prebuilt_root/darwin-arm64" ]] && { echo "darwin-arm64"; return; }
      [[ -d "$prebuilt_root/darwin-x86_64" ]] && { echo "darwin-x86_64"; return; }
      ;;
  esac
  echo "Unsupported Android NDK prebuilt host for $os/$arch under $prebuilt_root" >&2
  exit 1
}

cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
    return
  fi
  echo 4
}

abi_triple() {
  case "$1" in
    arm64-v8a) echo "aarch64-linux-android" ;;
    armeabi-v7a) echo "arm-linux-androideabi" ;;
    x86_64) echo "x86_64-linux-android" ;;
    x86) echo "i686-linux-android" ;;
    *)
      echo "Unsupported Android ABI: $1" >&2
      exit 1
      ;;
  esac
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ -d "$1" ]]; then
  NDK_ROOT="$(cd "$1" && pwd)"
  ANDROID_ABI="${2:-arm64-v8a}"
  ANDROID_API="${3:-30}"
else
  NDK_ROOT="$(resolve_ndk_root)"
  ANDROID_ABI="${1:-arm64-v8a}"
  ANDROID_API="${2:-30}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/third_party/mobilefinetuner/gemma_3_270m_lora_finetune"
BUILD_DIR="$SRC_DIR/build_android_${ANDROID_ABI}"
STAGE_DIR="$REPO_ROOT/outputs/android_gemma_trainer/${ANDROID_ABI}"

if [[ -d "$HOME/.cargo/bin" ]]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
if command -v rustup >/dev/null 2>&1; then
  rustup target add aarch64-linux-android
fi

mkdir -p "$BUILD_DIR" "$STAGE_DIR"

cmake \
  -S "$SRC_DIR" \
  -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ANDROID_ABI" \
  -DANDROID_PLATFORM="android-${ANDROID_API}" \
  -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEMMA_USE_HF_TOKENIZERS=ON \
  -DUSE_BLAS=OFF

cmake --build "$BUILD_DIR" -j"$(cpu_count)"

cp "$BUILD_DIR/train" "$STAGE_DIR/gemma_train"
if [[ -f "$BUILD_DIR/eval_mmlu_gemma" ]]; then
  cp "$BUILD_DIR/eval_mmlu_gemma" "$STAGE_DIR/"
fi

LIBCXX_SHARED="$NDK_ROOT/toolchains/llvm/prebuilt/$(host_tag)/sysroot/usr/lib/$(abi_triple "$ANDROID_ABI")/libc++_shared.so"
if [[ -f "$LIBCXX_SHARED" ]]; then
  cp "$LIBCXX_SHARED" "$STAGE_DIR/"
fi

echo "stage_dir=$STAGE_DIR"
