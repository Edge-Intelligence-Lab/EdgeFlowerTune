from __future__ import annotations

import argparse
import configparser
import io
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a git-free pinned gRPC source snapshot for Android builds"
    )
    root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--source-dir",
        default=str(root / "third_party" / "grpc-android-src"),
        help="Git-backed gRPC source tree",
    )
    parser.add_argument(
        "--dest-dir",
        default=str(root / "third_party" / "grpc-android-src-snapshot"),
        help="Output snapshot directory used for Android builds",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ignore_git(path: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name == ".git":
            ignored.add(name)
    return ignored


def gitlink_sha(source_dir: Path, submodule_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_dir), "ls-tree", "HEAD", submodule_path],
        check=True,
        capture_output=True,
        text=True,
    )
    parts = result.stdout.strip().split()
    if len(parts) < 3:
        raise RuntimeError(f"Failed to resolve gitlink SHA for {submodule_path}")
    return parts[2]


def github_codeload_url(remote_url: str, sha: str) -> str:
    url = remote_url.strip()
    if url.startswith("git@github.com:"):
        slug = url[len("git@github.com:") :]
    elif url.startswith("https://github.com/"):
        slug = url[len("https://github.com/") :]
    elif url.startswith("http://github.com/"):
        slug = url[len("http://github.com/") :]
    else:
        raise RuntimeError(f"Unsupported submodule remote URL: {remote_url}")
    if slug.endswith(".git"):
        slug = slug[:-4]
    return f"https://codeload.github.com/{slug}/tar.gz/{sha}"


def download_and_extract(url: str, dest_dir: Path) -> None:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = tar.getmembers()
        top_level: str | None = None
        for member in members:
            parts = Path(member.name).parts
            if not parts:
                continue
            if top_level is None:
                top_level = parts[0]
            if len(parts) == 1:
                continue
            relative = Path(*parts[1:])
            target = dest_dir / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            with target.open("wb") as handle:
                shutil.copyfileobj(file_obj, handle)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    dest_dir = Path(args.dest_dir).resolve()
    gitmodules = source_dir / ".gitmodules"
    if not source_dir.is_dir():
        raise RuntimeError(f"Missing source dir: {source_dir}")
    if not gitmodules.is_file():
        raise RuntimeError(f"Missing .gitmodules in {source_dir}")

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir, ignore=ignore_git)

    parser = configparser.ConfigParser()
    parser.read(gitmodules, encoding="utf-8")

    for section in parser.sections():
        submodule_path = parser.get(section, "path", fallback="").strip()
        remote_url = parser.get(section, "url", fallback="").strip()
        if not submodule_path or not remote_url:
            continue
        sha = gitlink_sha(source_dir, submodule_path)
        url = github_codeload_url(remote_url, sha)
        extract_dir = dest_dir / submodule_path
        print(f"[snapshot] {submodule_path} <- {sha}")
        download_and_extract(url, extract_dir)

    print(f"snapshot_dir={dest_dir}")


if __name__ == "__main__":
    main()
