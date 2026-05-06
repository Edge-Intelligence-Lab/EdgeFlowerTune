from __future__ import annotations

import argparse
import posixpath
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path

import paramiko


SYNC_ITEMS = [
    "src",
    "scripts",
    "configs",
    "clients",
    "docs",
    "requirements-classic-server.txt",
    "pyproject.toml",
    "README.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync the classic FedAvg+LoRA server tree to server3 and bootstrap its venv")
    parser.add_argument("--host", default="10.200.14.82")
    parser.add_argument("--username", default="AndyLu666")
    parser.add_argument("--remote-root", default="/home/AndyLu666/L-shaped-run-classic")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-venv", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def create_sync_archive() -> Path:
    root = repo_root()
    temp_dir = Path(tempfile.mkdtemp(prefix="lshaped_server3_sync_"))
    archive_path = temp_dir / "classic_fl_server3_sync.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in SYNC_ITEMS:
            local_path = root / item
            if not local_path.exists():
                raise RuntimeError(f"Missing sync item: {local_path}")
            tar.add(local_path, arcname=item)
    return archive_path


def connect(host: str, username: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        username=username,
        timeout=20,
        look_for_keys=True,
        allow_agent=True,
    )
    return client


def run_remote(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def ensure_success(client: paramiko.SSHClient, command: str) -> None:
    code, out, err = run_remote(client, command)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print(err, end="" if err.endswith("\n") else "\n")
    if code != 0:
        raise RuntimeError(f"Remote command failed with exit status {code}: {command}")


def upload_file(client: paramiko.SSHClient, local_path: Path, remote_path: str) -> None:
    sftp = client.open_sftp()
    try:
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def main() -> None:
    args = parse_args()
    remote_root = args.remote_root
    quoted_remote_root = shlex.quote(remote_root)
    remote_archive = "/tmp/classic_fl_server3_sync.tar.gz"
    archive_path: Path | None = None

    client = connect(args.host, args.username)
    try:
        if not args.skip_sync:
            archive_path = create_sync_archive()
            upload_file(client, archive_path, remote_archive)
            cleanup_targets = " ".join(
                shlex.quote(posixpath.join(remote_root, item))
                for item in SYNC_ITEMS
            )
            ensure_success(
                client,
                "set -e; "
                f"mkdir -p {quoted_remote_root}; "
                f"rm -rf {cleanup_targets}; "
                f"tar -xzf {shlex.quote(remote_archive)} -C {quoted_remote_root}; "
                f"rm -f {shlex.quote(remote_archive)}",
            )

        if not args.skip_venv:
            ensure_success(
                client,
                "set -e; "
                f"cd {quoted_remote_root}; "
                f"{shlex.quote(args.python)} -m venv .venv; "
                ". .venv/bin/activate; "
                "python --version",
            )

        if not args.skip_install:
            ensure_success(
                client,
                "set -e; "
                f"cd {quoted_remote_root}; "
                ". .venv/bin/activate; "
                "python -m pip install --upgrade pip setuptools wheel; "
                "python -m pip install -r requirements-classic-server.txt",
            )

        ensure_success(
            client,
            "set -e; "
            f"cd {quoted_remote_root}; "
            ". .venv/bin/activate; "
            "python - <<'PY'\n"
            "import flwr, numpy, yaml, safetensors, paramiko\n"
            "print('classic_server_env_ok')\n"
            "PY",
        )
        print(f"remote_root={remote_root}")
        print(f"remote_python={remote_root}/.venv/bin/python")
    finally:
        client.close()
        if archive_path is not None:
            shutil.rmtree(archive_path.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
