#!/usr/bin/env bash
set -euo pipefail

NDK_VERSION="${1:-r27d}"
INSTALL_ROOT="${2:-/datapool/BESTTOOLBOX/android}"
URL="https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip"
ZIP_PATH="${INSTALL_ROOT}/android-ndk-${NDK_VERSION}-linux.zip"
TARGET_DIR="${INSTALL_ROOT}/android-ndk-${NDK_VERSION}"

mkdir -p "$INSTALL_ROOT"
if [[ -d "$TARGET_DIR" ]]; then
  echo "ndk_root=$TARGET_DIR"
  exit 0
fi

curl -L --retry 5 --retry-delay 5 --connect-timeout 30 -o "$ZIP_PATH" "$URL"
if command -v unzip >/dev/null 2>&1; then
  unzip -q "$ZIP_PATH" -d "$INSTALL_ROOT"
else
  python3 - <<PY
import zipfile
from pathlib import Path
zip_path = Path(r"$ZIP_PATH")
target_root = Path(r"$INSTALL_ROOT")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(target_root)
PY
fi

find "$TARGET_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin" -type f -exec chmod +x {} +
find "$TARGET_DIR/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang" -type f -name 'clang*' -exec chmod +x {} + 2>/dev/null || true

echo "ndk_root=$TARGET_DIR"
