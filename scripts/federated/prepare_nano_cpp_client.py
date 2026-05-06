from __future__ import annotations

import argparse
import os
import posixpath
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path

import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap/sync/build the C++ Flower client on a Jetson Nano")
    parser.add_argument("--host", required=True, help="Nano host or IP")
    parser.add_argument("--username", default="jetson")
    parser.add_argument("--password", default=None, help="SSH password; defaults to env var if omitted")
    parser.add_argument("--password-env", default="NANO_PASSWORD", help="Environment variable used when --password is omitted")
    parser.add_argument("--remote-root", default=None, help="Remote repo root, default: /home/<username>/L-shaped")
    parser.add_argument("--backend", choices=("mock", "mft"), default="mock")
    parser.add_argument("--build-dir", default=None, help="Remote build dir, default depends on backend")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--model-bundle-dir",
        default=None,
        help="Optional local client-slim model bundle directory to upload to the Nano",
    )
    parser.add_argument(
        "--remote-model-dir",
        default=None,
        help="Remote model bundle directory, default: <remote-root>/models/<bundle-name>",
    )
    parser.add_argument(
        "--install-rust-if-missing",
        action="store_true",
        help="Install rustup/cargo on the Nano when building backend=mft and cargo is missing",
    )
    parser.add_argument(
        "--install-cmake-if-needed",
        action="store_true",
        help="Install a newer user-local cmake on the Nano when backend=mft requires it",
    )
    return parser.parse_args()


def resolve_password(args: argparse.Namespace) -> str:
    password = args.password or os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(
            f"Missing password: pass --password or set env var {args.password_env}"
        )
    return password


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def local_clients_dir() -> Path:
    path = repo_root() / "clients" / "cpp"
    if not path.is_dir():
        raise RuntimeError(f"Missing local client source tree: {path}")
    return path


def local_mft_ops_dir() -> Path:
    candidates = [
        repo_root() / "third_party" / "mobilefinetuner" / "operators",
        repo_root().parent / "operators",
    ]
    for path in candidates:
        if path.is_dir() and (path / "finetune_ops" / "core" / "tokenizer_hf.cpp").is_file():
            return path
    raise RuntimeError(f"Missing local MobileFineTuner operators tree in candidates: {candidates}")


def local_tokenizers_cpp_dir() -> Path:
    path = repo_root() / "third_party" / "tokenizers-cpp"
    if not path.is_dir():
        raise RuntimeError(f"Missing local vendored tokenizers-cpp tree: {path}")
    return path


def create_source_tarball(include_mft: bool) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="lshaped_nano_sync_"))
    archive_path = temp_dir / "cpp_client_sources.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(local_clients_dir(), arcname="clients/cpp")
        if include_mft:
            tar.add(local_mft_ops_dir(), arcname="third_party/mobilefinetuner/operators")
            tar.add(local_tokenizers_cpp_dir(), arcname="third_party/tokenizers-cpp")
    return archive_path


def create_directory_tarball(source_dir: Path, prefix: str) -> Path:
    if not source_dir.is_dir():
        raise RuntimeError(f"Expected directory, got: {source_dir}")
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    archive_path = temp_dir / f"{source_dir.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)
    return archive_path


def connect(host: str, username: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        username=username,
        password=password,
        timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def run_remote(client: paramiko.SSHClient, command: str) -> None:
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print(err, end="" if err.endswith("\n") else "\n")
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        raise RuntimeError(f"Remote command failed with exit status {exit_status}: {command}")


def upload_file(client: paramiko.SSHClient, local_path: Path, remote_path: str) -> None:
    sftp = client.open_sftp()
    try:
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def main() -> None:
    args = parse_args()
    password = resolve_password(args)
    remote_root = args.remote_root or f"/home/{args.username}/L-shaped"
    build_dir = args.build_dir or (
        "build/cpp_client_mft" if args.backend == "mft" else "build/cpp_client_mock"
    )
    enable_mft = "ON" if args.backend == "mft" else "OFF"
    include_mft = args.backend == "mft"
    quoted_password = shlex.quote(password)
    quoted_remote_root = shlex.quote(remote_root)
    quoted_build_dir = shlex.quote(build_dir)

    source_archive = None
    model_archive = None
    model_bundle_name = None
    remote_model_dir = args.remote_model_dir
    client = None
    try:
        if not args.skip_sync:
            source_archive = create_source_tarball(include_mft=include_mft)
            if args.model_bundle_dir:
                bundle_dir = Path(args.model_bundle_dir).resolve()
                model_archive = create_directory_tarball(bundle_dir, "lshaped_model_sync_")
                model_bundle_name = bundle_dir.name
                if not remote_model_dir:
                    remote_model_dir = posixpath.join(remote_root, "models", bundle_dir.name)

        client = connect(args.host, args.username, password)
        print(f"[nano] connected to {args.host} as {args.username}")

        if not args.skip_install:
            run_remote(
                client,
                "set -e; "
                f"echo {quoted_password} | sudo -S apt-get update; "
                f"echo {quoted_password} | sudo -S DEBIAN_FRONTEND=noninteractive apt-get install -y "
                "libgrpc++-dev protobuf-compiler-grpc curl build-essential pkg-config cmake",
            )

        if include_mft and args.install_rust_if_missing:
            run_remote(
                client,
                "set -e; "
                "if ! command -v cargo >/dev/null 2>&1; then "
                "curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal; "
                "fi; "
                "export PATH=\"$HOME/.cargo/bin:$PATH\"; "
                "rustup set profile minimal; "
                "cargo --version; rustc --version",
            )

        if include_mft and args.install_cmake_if_needed:
            run_remote(
                client,
                "set -e; "
                "python3 -m pip install --user --upgrade 'cmake>=3.27,<4'; "
                "export PATH=\"$HOME/.local/bin:$PATH\"; "
                "cmake --version | head -n 1",
            )

        if not args.skip_sync:
            remote_source_archive = "/tmp/lshaped_cpp_client_sources.tar.gz"
            upload_file(client, source_archive, remote_source_archive)
            remote_clients_cpp = posixpath.join(remote_root, "clients", "cpp")
            remote_mft_ops = posixpath.join(remote_root, "third_party", "mobilefinetuner", "operators")
            remote_tokenizers_cpp = posixpath.join(remote_root, "third_party", "tokenizers-cpp")
            quoted_source_archive = shlex.quote(remote_source_archive)
            quoted_remote_clients_cpp = shlex.quote(remote_clients_cpp)
            quoted_remote_mft_ops = shlex.quote(remote_mft_ops)
            quoted_remote_tokenizers_cpp = shlex.quote(remote_tokenizers_cpp)
            run_remote(
                client,
                "set -e; "
                f"mkdir -p {quoted_remote_root}; "
                f"rm -rf {quoted_remote_clients_cpp}; "
                + (
                    f"rm -rf {quoted_remote_mft_ops} {quoted_remote_tokenizers_cpp}; "
                    if include_mft
                    else ""
                )
                + f"tar -xzf {quoted_source_archive} -C {quoted_remote_root}; "
                f"rm -f {quoted_source_archive}",
            )

            if model_archive is not None:
                assert remote_model_dir is not None
                assert model_bundle_name is not None
                remote_model_archive = "/tmp/lshaped_client_model_bundle.tar.gz"
                upload_file(client, model_archive, remote_model_archive)
                remote_model_parent = posixpath.dirname(remote_model_dir)
                remote_model_name = posixpath.basename(remote_model_dir)
                extracted_model_dir = posixpath.join(remote_model_parent, model_bundle_name)
                quoted_remote_model_archive = shlex.quote(remote_model_archive)
                quoted_remote_model_parent = shlex.quote(remote_model_parent)
                quoted_remote_model_dir = shlex.quote(remote_model_dir)
                quoted_extracted_model_dir = shlex.quote(extracted_model_dir)
                run_remote(
                    client,
                    "set -e; "
                    f"mkdir -p {quoted_remote_model_parent}; "
                    f"rm -rf {quoted_remote_model_dir}; "
                    f"tar -xzf {quoted_remote_model_archive} -C {quoted_remote_model_parent}; "
                    + (
                        f"mv {quoted_extracted_model_dir} {quoted_remote_model_dir}; "
                        if extracted_model_dir != remote_model_dir
                        else ""
                    )
                    + f"rm -f {quoted_remote_model_archive}",
                )

        if not args.skip_build:
            build_prefix = ""
            if include_mft:
                build_prefix = "export PATH=\"$HOME/.local/bin:$HOME/.cargo/bin:$PATH\"; "
            run_remote(
                client,
                "set -e; "
                + build_prefix
                + f"cd {quoted_remote_root}; "
                f"cmake -S clients/cpp -B {quoted_build_dir} "
                f"-DLSHAPED_ENABLE_MFT={enable_mft} -DLSHAPED_REGENERATE_PROTO=ON; "
                f"cmake --build {quoted_build_dir} -j{args.jobs}",
            )

        if remote_model_dir:
            print(f"[nano] remote_model_dir={remote_model_dir}")

    finally:
        if client is not None:
            client.close()
        if source_archive is not None:
            shutil.rmtree(source_archive.parent, ignore_errors=True)
        if model_archive is not None:
            shutil.rmtree(model_archive.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
