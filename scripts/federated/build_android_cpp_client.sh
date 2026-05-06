#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <android-ndk-root> [mock|mft] [arm64-v8a] [30]" >&2
  echo "   or: $0 [mock|mft] [arm64-v8a] [30]    # auto-detect NDK root" >&2
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
      if [[ -d "$prebuilt_root/linux-x86_64" ]]; then
        echo "linux-x86_64"
        return
      fi
      ;;
    Darwin)
      if [[ "$arch" == "arm64" && -d "$prebuilt_root/darwin-arm64" ]]; then
        echo "darwin-arm64"
        return
      fi
      if [[ -d "$prebuilt_root/darwin-x86_64" ]]; then
        echo "darwin-x86_64"
        return
      fi
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

grpc_source_is_usable() {
  local source_dir="$1"
  [[ -n "$source_dir" && -d "$source_dir" && -f "$source_dir/CMakeLists.txt" ]] || return 1
  local boringssl_root="$source_dir/third_party/boringssl-with-bazel"
  if [[ -d "$boringssl_root" ]]; then
    if [[ ! -f "$boringssl_root/gen/sources.cmake" && ! -f "$boringssl_root/src/sources.cmake" ]]; then
      return 1
    fi
  fi
  if [[ -d "$source_dir/.git" || -f "$source_dir/.git" ]]; then
    if command -v git >/dev/null 2>&1; then
      if git -C "$source_dir" submodule status --recursive 2>/dev/null | grep -Eq '^[+-U]'; then
        return 1
      fi
    fi
  fi
  return 0
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ -d "$1" ]]; then
  NDK_ROOT="$(cd "$1" && pwd)"
  BACKEND="${2:-mft}"
  ANDROID_ABI="${3:-arm64-v8a}"
  ANDROID_API="${4:-30}"
else
  NDK_ROOT="$(resolve_ndk_root)"
  BACKEND="${1:-mft}"
  ANDROID_ABI="${2:-arm64-v8a}"
  ANDROID_API="${3:-30}"
fi

if [[ "$BACKEND" != "mock" && "$BACKEND" != "mft" ]]; then
  echo "Unsupported backend: $BACKEND" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/android_${ANDROID_ABI}_${BACKEND}"
STAGE_DIR="$REPO_ROOT/outputs/android_client/${ANDROID_ABI}/${BACKEND}"
ENABLE_MFT=OFF
DEFAULT_BUNDLED_GRPC_SOURCE_DIR="$REPO_ROOT/third_party/grpc-android-src"
if ! grpc_source_is_usable "$DEFAULT_BUNDLED_GRPC_SOURCE_DIR"; then
  SNAPSHOT_GRPC_SOURCE_DIR="$REPO_ROOT/third_party/grpc-android-src-snapshot"
  if grpc_source_is_usable "$SNAPSHOT_GRPC_SOURCE_DIR"; then
    DEFAULT_BUNDLED_GRPC_SOURCE_DIR="$SNAPSHOT_GRPC_SOURCE_DIR"
  fi
fi
BUNDLED_GRPC_SOURCE_DIR="${LSHAPED_BUNDLED_GRPC_SOURCE_DIR-$DEFAULT_BUNDLED_GRPC_SOURCE_DIR}"
BUNDLED_GRPC_GIT_TAG="${LSHAPED_BUNDLED_GRPC_GIT_TAG:-v1.68.0}"
REGENERATE_PROTO="${LSHAPED_REGENERATE_PROTO:-OFF}"
ALLOW_DIRTY_VENDORED_GRPC="${LSHAPED_ALLOW_DIRTY_VENDORED_GRPC:-0}"
if [[ -n "${LSHAPED_CMAKE_GENERATOR:-}" ]]; then
  CMAKE_GENERATOR="$LSHAPED_CMAKE_GENERATOR"
elif command -v ninja >/dev/null 2>&1; then
  CMAKE_GENERATOR="Ninja"
else
  CMAKE_GENERATOR="Unix Makefiles"
fi
PYTHON_BIN="${LSHAPED_PYTHON_BIN:-}"
if [[ -d "$HOME/.cargo/bin" ]]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
export GIT_HTTP_VERSION="${LSHAPED_GIT_HTTP_VERSION:-HTTP/1.1}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Missing python interpreter for Android build helper patching" >&2
    exit 1
  fi
fi

if [[ "$BACKEND" == "mft" ]]; then
  ENABLE_MFT=ON
  if command -v rustup >/dev/null 2>&1; then
    rustup target add aarch64-linux-android
  fi
fi

if [[ -n "$BUNDLED_GRPC_SOURCE_DIR" && ! -d "$BUNDLED_GRPC_SOURCE_DIR" ]]; then
  BUNDLED_GRPC_SOURCE_DIR=""
fi

if [[ -n "$BUNDLED_GRPC_SOURCE_DIR" && "$ALLOW_DIRTY_VENDORED_GRPC" != "1" ]]; then
  if ! grpc_source_is_usable "$BUNDLED_GRPC_SOURCE_DIR"; then
    echo "Vendored gRPC source is not a clean, pinned checkout; falling back to upstream fetch." >&2
    BUNDLED_GRPC_SOURCE_DIR=""
  fi
fi

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cached_grpc_source_dir="$(sed -n 's/^LSHAPED_BUNDLED_GRPC_SOURCE_DIR:PATH=//p' "$BUILD_DIR/CMakeCache.txt" | head -n 1)"
  cached_grpc_git_tag="$(sed -n 's/^LSHAPED_BUNDLED_GRPC_GIT_TAG:STRING=//p' "$BUILD_DIR/CMakeCache.txt" | head -n 1)"
  if [[ "$cached_grpc_source_dir" != "$BUNDLED_GRPC_SOURCE_DIR" || "$cached_grpc_git_tag" != "$BUNDLED_GRPC_GIT_TAG" ]]; then
    rm -rf "$BUILD_DIR"
  fi
fi

mkdir -p "$BUILD_DIR" "$STAGE_DIR"

if [[ -n "$BUNDLED_GRPC_SOURCE_DIR" ]] && [[ "$ANDROID_ABI" == "arm64-v8a" || "$ANDROID_ABI" == "armeabi-v7a" || "$ANDROID_ABI" == "x86_64" || "$ANDROID_ABI" == "x86" ]]; then
  ABSL_ATTRS_H="$BUNDLED_GRPC_SOURCE_DIR/third_party/abseil-cpp/absl/base/attributes.h"
  if [[ -f "$ABSL_ATTRS_H" ]]; then
    "$PYTHON_BIN" - "$ABSL_ATTRS_H" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
warn_unused = (
    "// ABSL_ATTRIBUTE_WARN_UNUSED\\n"
    "//\\n"
    "// Marks a type or function result as requiring use by the caller. Newer\\n"
    "// protobuf releases expect this macro, but the vendored Abseil snapshot in\\n"
    "// this tree predates it.\\n"
)
if "ABSL_ATTRIBUTE_WARN_UNUSED" not in text:
    needle = "// ABSL_ATTRIBUTE_NONNULL\\n"
    replacement = (
        warn_unused +
        "#if ABSL_HAVE_CPP_ATTRIBUTE(nodiscard)\\n"
        "#define ABSL_ATTRIBUTE_WARN_UNUSED [[nodiscard]]\\n"
        "#elif ABSL_HAVE_ATTRIBUTE(warn_unused_result) || \\\\\\n"
        "    (defined(__GNUC__) && !defined(__clang__))\\n"
        "#define ABSL_ATTRIBUTE_WARN_UNUSED __attribute__((warn_unused_result))\\n"
        "#else\\n"
        "#define ABSL_ATTRIBUTE_WARN_UNUSED\\n"
        "#endif\\n\\n"
        "// ABSL_ATTRIBUTE_VIEW\\n"
        "//\\n"
        "// Newer protobuf headers annotate iterator/view-like classes with this macro.\\n"
        "#ifndef ABSL_ATTRIBUTE_VIEW\\n"
        "#define ABSL_ATTRIBUTE_VIEW\\n"
        "#endif\\n\\n"
        "// ABSL_NULLABILITY_COMPATIBLE\\n"
        "//\\n"
        "// Newer protobuf headers annotate smart-pointer-like classes with this macro.\\n"
        "#ifndef ABSL_NULLABILITY_COMPATIBLE\\n"
        "#if ABSL_HAVE_ATTRIBUTE(nullability_compatible)\\n"
        "#define ABSL_NULLABILITY_COMPATIBLE __attribute__((nullability_compatible))\\n"
        "#else\\n"
        "#define ABSL_NULLABILITY_COMPATIBLE\\n"
        "#endif\\n"
        "#endif\\n\\n"
        "// ABSL_ATTRIBUTE_NONNULL\\n"
    )
    if needle in text:
        text = text.replace(needle, replacement, 1)

if "ABSL_DEPRECATE_AND_INLINE" not in text:
    needle = "#endif\\n\\n// When deprecating Abseil code, it is sometimes necessary to turn off the\\n"
    replacement = (
        "#endif\\n\\n"
        "// ABSL_DEPRECATE_AND_INLINE()\\n"
        "//\\n"
        "// Newer protobuf headers use this compatibility helper, but this vendored\\n"
        "// Abseil snapshot predates the macro.\\n"
        "#ifndef ABSL_DEPRECATE_AND_INLINE\\n"
        "#define ABSL_DEPRECATE_AND_INLINE() [[deprecated]]\\n"
        "#endif\\n\\n"
        "// When deprecating Abseil code, it is sometimes necessary to turn off the\\n"
    )
    if needle in text:
        text = text.replace(needle, replacement, 1)

path.write_text(text, encoding="utf-8")
PY
  fi
  PORT_DEF_INC="$BUNDLED_GRPC_SOURCE_DIR/third_party/protobuf/src/google/protobuf/port_def.inc"
  if [[ -f "$PORT_DEF_INC" ]]; then
    "$PYTHON_BIN" - "$PORT_DEF_INC" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = "elif !defined(__CYGWIN__) && !defined(__MINGW32__) &&                 \\\n"
replacement = "elif !defined(__CYGWIN__) && !defined(__MINGW32__) && !defined(__ANDROID__) &&                 \\\n"
if needle in text and replacement not in text:
    text = text.replace(needle, replacement, 1)
needle = "#define PROTOBUF_FUTURE_ADD_EARLY_WARN_UNUSED ABSL_ATTRIBUTE_WARN_UNUSED\n"
replacement = (
    "#if defined(__ANDROID__)\n"
    "#define PROTOBUF_FUTURE_ADD_EARLY_WARN_UNUSED\n"
    "#else\n"
    "#define PROTOBUF_FUTURE_ADD_EARLY_WARN_UNUSED ABSL_ATTRIBUTE_WARN_UNUSED\n"
    "#endif\n"
)
if needle in text and replacement not in text:
    text = text.replace(needle, replacement, 1)
constinit_block_pattern = re.compile(
    r"#ifdef PROTOBUF_CONSTINIT\n"
    r"#error PROTOBUF_CONSTINIT was previously defined\n"
    r"#endif\n\n"
    r"(?:#ifdef PROTOBUF_CONSTEXPR\n"
    r"#error PROTOBUF_CONSTEXPR was previously defined\n"
    r"#endif\n\n)?"
    r"#if defined\(PROTOBUF_USE_DLLS\) && \(defined\(_WIN32\) \|\| defined\(__CYGWIN__\)\)\n"
    r".*?"
    r"#ifndef PROTOBUF_CONSTINIT\n"
    r"#define PROTOBUF_CONSTINIT\n"
    r"(?:#define PROTOBUF_CONSTEXPR(?: constexpr)?\n)?"
    r"#endif\n",
    re.S,
)
constinit_block_span_pattern = re.compile(
    r"#ifdef PROTOBUF_CONSTINIT\n"
    r"#error PROTOBUF_CONSTINIT was previously defined\n"
    r"#endif\n\n"
    r".*?"
    r"(?=\n// Some globals with an empty non-trivial destructor are annotated with\n)",
    re.S,
)
canonical_constinit_block = """#ifdef PROTOBUF_CONSTINIT
#error PROTOBUF_CONSTINIT was previously defined
#endif

#ifdef PROTOBUF_CONSTEXPR
#error PROTOBUF_CONSTEXPR was previously defined
#endif

#if defined(PROTOBUF_USE_DLLS) && (defined(_WIN32) || defined(__CYGWIN__))
// On Windows, `constinit` structures cannot include pointers to dllimport
// symbols, which are defined in another dll. If libprotobuf is built as a dll,
// generated parse tables breaks this. See
// https://github.com/protocolbuffers/protobuf/issues/10159 and
// https://github.com/protocolbuffers/protobuf/pull/13240. Work around this by
// suppressing `constinit`.
#define PROTOBUF_CONSTINIT
#define PROTOBUF_CONSTEXPR constexpr
#elif defined(_MSC_VER) && !defined(__clang__)
// On MSVC, but not clang-cl, the above workaround must extend to static library
// builds too. MSVC can avoid a global constructor when initializing structures
// containing pointers to same-dll symbols, it relies on the optimizer for this,
// so we can't enforce constinit. This limitation does not apply to constexpr.
// See https://godbolt.org/z/hsT9e3zs4
#define PROTOBUF_CONSTINIT
#define PROTOBUF_CONSTEXPR constexpr
#elif defined(__ANDROID__)
// Android NDK Clang rejects several protobuf globals under the stricter
// constinit checks enabled below, so disable the annotation there.
#define PROTOBUF_CONSTINIT
#define PROTOBUF_CONSTEXPR constexpr
#elif defined(__GNUC__) && !defined(__clang__)
// GCC doesn't support constinit aggregate initialization of absl::Cord.
#define PROTOBUF_CONSTINIT
#define PROTOBUF_CONSTEXPR constexpr
#elif defined(__cpp_constinit)
#define PROTOBUF_CONSTINIT constinit
#define PROTOBUF_CONSTEXPR constexpr
#define PROTOBUF_CONSTINIT_DEFAULT_INSTANCES
// Some older Clang versions incorrectly raise an error about
// constant-initializing weak default instance pointers. Versions 12.0 and
// higher seem to work, except that XCode 12.5.1 shows the error even though it
// uses Clang 12.0.5.
#elif ABSL_HAVE_CPP_ATTRIBUTE(clang::require_constant_initialization) && \\
    ((defined(__APPLE__) && PROTOBUF_CLANG_MIN(13, 0)) ||                \\
     (!defined(__APPLE__) && PROTOBUF_CLANG_MIN(12, 0)))
#define PROTOBUF_CONSTINIT [[clang::require_constant_initialization]]
#define PROTOBUF_CONSTEXPR constexpr
#define PROTOBUF_CONSTINIT_DEFAULT_INSTANCES
#else
#define PROTOBUF_CONSTINIT
#define PROTOBUF_CONSTEXPR
#endif
"""
text, count = constinit_block_pattern.subn(canonical_constinit_block, text, count=1)
if count == 0:
    text, count = constinit_block_span_pattern.subn(canonical_constinit_block, text, count=1)
already_canonical = (
    "#ifdef PROTOBUF_CONSTEXPR\n#error PROTOBUF_CONSTEXPR was previously defined\n#endif\n" in text
    and "#elif defined(__ANDROID__)\n// Android NDK Clang rejects several protobuf globals under the stricter\n// constinit checks enabled below, so disable the annotation there.\n#define PROTOBUF_CONSTINIT\n#define PROTOBUF_CONSTEXPR constexpr\n" in text
    and "#ifndef PROTOBUF_CONSTEXPR\n#define PROTOBUF_CONSTEXPR\n#endif\n" in text
)
if count == 0 and not already_canonical:
    raise SystemExit(f"failed to canonicalize PROTOBUF_CONSTINIT block in {path}")
path.write_text(text, encoding="utf-8")
PY
  fi
  BASIC_SEQ_H="$BUNDLED_GRPC_SOURCE_DIR/src/core/lib/promise/detail/basic_seq.h"
  if [[ -f "$BASIC_SEQ_H" ]]; then
    "$PYTHON_BIN" - "$BASIC_SEQ_H" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = "Construct(&state_,\n                    Traits::template CallSeqFactory(f_, *cur_, std::move(arg)));"
new = "Construct(&state_,\n                    Traits::CallSeqFactory(f_, *cur_, std::move(arg)));"
if old in text and new not in text:
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
PY
  fi
  SSL_TRANSPORT_SECURITY_CC="$BUNDLED_GRPC_SOURCE_DIR/src/core/tsi/ssl_transport_security.cc"
  if [[ -f "$SSL_TRANSPORT_SECURITY_CC" ]]; then
    "$PYTHON_BIN" - "$SSL_TRANSPORT_SECURITY_CC" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
replacements = [
    (
        "    X509_STORE* cert_store = SSL_CTX_get_cert_store(impl->ssl_context);\n"
        "    X509_STORE_set_get_crl(cert_store, GetCrlFromProvider);\n"
        "    X509_STORE_set_check_crl(cert_store, CheckCrlPassthrough);\n"
        "    X509_STORE_set_verify_cb(cert_store, verify_cb);\n"
        "    X509_VERIFY_PARAM* param = X509_STORE_get0_param(cert_store);\n"
        "    X509_VERIFY_PARAM_set_flags(\n"
        "        param, X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL);\n",
        "    X509_STORE* cert_store = SSL_CTX_get_cert_store(impl->ssl_context);\n"
        "#if !defined(OPENSSL_IS_BORINGSSL)\n"
        "    X509_STORE_set_get_crl(cert_store, GetCrlFromProvider);\n"
        "    X509_STORE_set_check_crl(cert_store, CheckCrlPassthrough);\n"
        "    X509_STORE_set_verify_cb(cert_store, verify_cb);\n"
        "    X509_VERIFY_PARAM* param = X509_STORE_get0_param(cert_store);\n"
        "    X509_VERIFY_PARAM_set_flags(\n"
        "        param, X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL);\n"
        "#else\n"
        "    gpr_log(GPR_ERROR, \"CRL provider is unsupported with BoringSSL.\");\n"
        "#endif\n",
    ),
    (
        "        X509_STORE* cert_store = SSL_CTX_get_cert_store(impl->ssl_contexts[i]);\n"
        "        X509_STORE_set_get_crl(cert_store, GetCrlFromProvider);\n"
        "        X509_STORE_set_check_crl(cert_store, CheckCrlPassthrough);\n"
        "        X509_STORE_set_verify_cb(cert_store, verify_cb);\n"
        "        X509_VERIFY_PARAM* param = X509_STORE_get0_param(cert_store);\n"
        "        X509_VERIFY_PARAM_set_flags(\n"
        "            param, X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL);\n",
        "        X509_STORE* cert_store = SSL_CTX_get_cert_store(impl->ssl_contexts[i]);\n"
        "#if !defined(OPENSSL_IS_BORINGSSL)\n"
        "        X509_STORE_set_get_crl(cert_store, GetCrlFromProvider);\n"
        "        X509_STORE_set_check_crl(cert_store, CheckCrlPassthrough);\n"
        "        X509_STORE_set_verify_cb(cert_store, verify_cb);\n"
        "        X509_VERIFY_PARAM* param = X509_STORE_get0_param(cert_store);\n"
        "        X509_VERIFY_PARAM_set_flags(\n"
        "            param, X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL);\n"
        "#else\n"
        "        gpr_log(GPR_ERROR, \"CRL provider is unsupported with BoringSSL.\");\n"
        "#endif\n",
    ),
]
for old, new in replacements:
    if old in text and new not in text:
        text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
PY
  fi
  RE2_DFA_CC="$BUNDLED_GRPC_SOURCE_DIR/third_party/re2/re2/dfa.cc"
  if [[ -f "$RE2_DFA_CC" ]]; then
    "$PYTHON_BIN" - "$RE2_DFA_CC" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
replacements = {
    "absl::MutexLock l(mutex_);": "absl::MutexLock l(&mutex_);",
    "absl::MutexLock l(dfa_->mutex_);": "absl::MutexLock l(&dfa_->mutex_);",
    "absl::MutexLock lock(mutex_);": "absl::MutexLock lock(&mutex_);",
}
for old, new in replacements.items():
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
PY
  fi
  ZLIB_CMAKELISTS="$BUNDLED_GRPC_SOURCE_DIR/third_party/zlib/CMakeLists.txt"
  if [[ -f "$ZLIB_CMAKELISTS" ]]; then
    "$PYTHON_BIN" - "$ZLIB_CMAKELISTS" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
shared_decl = "add_library(zlib SHARED ${ZLIB_SRCS} ${ZLIB_DLL_SRCS} ${ZLIB_PUBLIC_HDRS} ${ZLIB_PRIVATE_HDRS})"
static_decl = "add_library(zlib STATIC ${ZLIB_SRCS} ${ZLIB_DLL_SRCS} ${ZLIB_PUBLIC_HDRS} ${ZLIB_PRIVATE_HDRS})"
if shared_decl in text and static_decl not in text:
    text = text.replace(shared_decl, static_decl, 1)
old_unix_block = (
    "if(UNIX)\n"
    "    # On unix-like platforms the library is almost always called libz\n"
    "   set_target_properties(zlib zlibstatic PROPERTIES OUTPUT_NAME z)\n"
    "   if(NOT APPLE)\n"
    "     set_target_properties(zlib PROPERTIES LINK_FLAGS \"-Wl,--version-script,\\\"${CMAKE_CURRENT_SOURCE_DIR}/zlib.map\\\"\")\n"
    "   endif()\n"
    "elseif(BUILD_SHARED_LIBS AND WIN32)\n"
)
new_unix_block = (
    "if(UNIX)\n"
    "    # On unix-like platforms the library is almost always called libz\n"
    "   if(ANDROID)\n"
    "     set_target_properties(zlibstatic PROPERTIES OUTPUT_NAME z)\n"
    "   else()\n"
    "     set_target_properties(zlib zlibstatic PROPERTIES OUTPUT_NAME z)\n"
    "     if(NOT APPLE)\n"
    "       set_target_properties(zlib PROPERTIES LINK_FLAGS \"-Wl,--version-script,\\\"${CMAKE_CURRENT_SOURCE_DIR}/zlib.map\\\"\")\n"
    "     endif()\n"
    "   endif()\n"
    "elseif(BUILD_SHARED_LIBS AND WIN32)\n"
)
if old_unix_block in text and new_unix_block not in text:
    text = text.replace(old_unix_block, new_unix_block, 1)
path.write_text(text, encoding="utf-8")
PY
  fi
fi

cmake_args=(
  -S "$REPO_ROOT/clients/cpp"
  -B "$BUILD_DIR"
  -G "$CMAKE_GENERATOR"
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake"
  -DANDROID_ABI="$ANDROID_ABI"
  -DANDROID_PLATFORM="android-${ANDROID_API}"
  -DANDROID_STL=c++_shared
  -DLSHAPED_ENABLE_MFT="$ENABLE_MFT"
  -DLSHAPED_ENABLE_BUNDLED_GRPC=ON
  -DLSHAPED_REGENERATE_PROTO="$REGENERATE_PROTO"
  -DLSHAPED_BUNDLED_GRPC_GIT_TAG="$BUNDLED_GRPC_GIT_TAG"
  -DBUILD_TESTING=OFF
  -DZLIB_BUILD_TESTING=OFF
  -DBENCHMARK_ENABLE_TESTING=OFF
  -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
)

if [[ -n "$BUNDLED_GRPC_SOURCE_DIR" ]]; then
  cmake_args+=(-DLSHAPED_BUNDLED_GRPC_SOURCE_DIR="$BUNDLED_GRPC_SOURCE_DIR")
fi

cmake "${cmake_args[@]}"

cmake --build "$BUILD_DIR" -j"$(cpu_count)"

cp "$BUILD_DIR/lshaped_flower_client" "$STAGE_DIR/"

LIBCXX_SHARED="$NDK_ROOT/toolchains/llvm/prebuilt/$(host_tag)/sysroot/usr/lib/$(abi_triple "$ANDROID_ABI")/libc++_shared.so"
if [[ -f "$LIBCXX_SHARED" ]]; then
  cp "$LIBCXX_SHARED" "$STAGE_DIR/"
fi

echo "stage_dir=$STAGE_DIR"
