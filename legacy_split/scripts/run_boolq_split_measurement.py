from __future__ import annotations

import argparse
import csv
import json
import os
import posixpath
import re
import shlex
import statistics
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko
import yaml

MAX_PHONE_AVG_POWER_W = 20.0
MAX_PHONE_UID_POWER_W = 10.0
MIN_RELIABLE_PHONE_POWER_WINDOW_SEC = 300.0
POWER_WINDOW_EXTRA_SEC = 1.0
POWER_WINDOW_TOLERANCE_SEC = 0.5


@dataclass
class AndroidSpec:
    client_id: str
    serial: str
    device_root: str


@dataclass
class JetsonSpec:
    client_id: str
    host: str
    username: str
    password: str
    remote_root: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SplitLoRA one-round measurement")
    parser.add_argument(
        "--base-config",
        default="L-shaped_code_docs_backup/legacy_split/configs/splitlora_gemma270m_eight_client_boolq_seq64_b8_r1_l3.yaml",
    )
    parser.add_argument(
        "--client-specs-json",
        default="L-shaped_code_docs_backup/legacy_split/configs/eight_clients_mixed_split_mft.json",
    )
    parser.add_argument(
        "--prepare-boolq-script",
        default="L-shaped_code_docs_backup/scripts/prepare_boolq_mmlu_csv.py",
        help="Dataset preparation script kept for backward compatibility",
    )
    parser.add_argument(
        "--prepare-dataset-script",
        default="",
        help="Optional dataset preparation script. Overrides --prepare-boolq-script when set.",
    )
    parser.add_argument("--skip-prepare-script", action="store_true")
    parser.add_argument(
        "--prepare-script-extra-args",
        nargs="*",
        default=[],
        help="Extra arguments forwarded to the local dataset preparation script",
    )
    parser.add_argument("--prepare-script-output-dir", default="")
    parser.add_argument("--prepare-script-model-dir", default="")
    parser.add_argument("--prepare-script-seq-len", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-label", default="gemma3_boolq_split_measurement")
    parser.add_argument("--adb-path", default="")
    parser.add_argument("--server-port", type=int, default=19174)
    parser.add_argument("--server-ssh-host", default="10.200.14.82")
    parser.add_argument("--server-ssh-username", default="AndyLu666")
    parser.add_argument("--server-remote-root", default="/home/AndyLu666/L-shaped-run-classic")
    parser.add_argument("--server-address-host", default="10.200.14.82")
    parser.add_argument("--server-python", default="/home/AndyLu666/gemma3_server_eval_env/bin/python")
    parser.add_argument("--cuda-visible-devices", default="4")
    parser.add_argument("--server-wait-timeout", type=int, default=180)
    parser.add_argument("--server-exit-timeout", type=int, default=3600)
    parser.add_argument("--client-exit-timeout", type=int, default=1800)
    parser.add_argument("--connect-max-attempts", type=int, default=60)
    parser.add_argument("--connect-ready-timeout-ms", type=int, default=15000)
    parser.add_argument("--connect-retry-delay-ms", type=int, default=5000)
    parser.add_argument("--timeout-sec", type=int, default=10800)
    parser.add_argument("--default-nano-password", default="jetson")
    parser.add_argument("--default-nano-remote-root", default="/home/jetson/L-shaped")
    parser.add_argument("--no-phone-keep-awake", action="store_true")
    parser.add_argument("--skip-android-binary-push", action="store_true")
    parser.add_argument("--skip-android-model-push", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_local_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run_local(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 7200,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        if check:
            raise
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        return subprocess.CompletedProcess(cmd, 124, stdout=stdout, stderr=stderr)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def log(message: str) -> None:
    print(message, flush=True)


def resolve_adb_path(raw_path: str) -> str:
    if raw_path:
        return str(Path(raw_path).expanduser().resolve())
    which_adb = shutil_which("adb")
    if which_adb:
        return which_adb
    sdk_adb = Path.home() / "Library" / "Android" / "sdk" / "platform-tools" / "adb"
    if sdk_adb.is_file():
        return str(sdk_adb.resolve())
    return "adb"


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def adb(adb_bin: str, serial: str, args: list[str], *, timeout: int = 1200, check: bool = True) -> str:
    proc = run_local([adb_bin, "-s", serial, *args], timeout=timeout, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"ADB command failed ({proc.returncode}): {adb_bin} -s {serial} {' '.join(args)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def adb_shell(adb_bin: str, serial: str, command: str, *, timeout: int = 1200, check: bool = True) -> str:
    return adb(adb_bin, serial, ["shell", command], timeout=timeout, check=check)


def connect_keyed(host: str, username: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, timeout=30, look_for_keys=True, allow_agent=True)
    return client


def connect_password(host: str, username: str, password: str) -> paramiko.SSHClient:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                host,
                username=username,
                password=password,
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
                look_for_keys=False,
                allow_agent=False,
            )
            return client
        except Exception as exc:
            last_error = exc
            try:
                client.close()
            except Exception:
                pass
            if attempt == 5:
                break
            time.sleep(min(5 * attempt, 15))
    assert last_error is not None
    raise last_error


def run_remote(client: paramiko.SSHClient, command: str, *, get_pty: bool = True) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, get_pty=get_pty)
    del stdin
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def run_remote_sudo(
    client: paramiko.SSHClient,
    password: str,
    command: str,
    *,
    get_pty: bool = True,
) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(
        f"sudo -S -p '' bash -lc {shlex.quote(command)}",
        get_pty=get_pty,
    )
    stdin.write(password + "\n")
    stdin.flush()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def load_specs(path: Path) -> tuple[list[AndroidSpec], list[JetsonSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    android_specs: list[AndroidSpec] = []
    jetson_specs: list[JetsonSpec] = []
    for item in raw:
        if str(item.get("type", "")).strip().lower() == "android":
            android_specs.append(
                AndroidSpec(
                    client_id=str(item["client_id"]),
                    serial=str(item["serial"]),
                    device_root=str(item.get("device_root", "/data/local/tmp/L-shaped")),
                )
            )
        elif str(item.get("type", "")).strip().lower() == "nano":
            jetson_specs.append(
                JetsonSpec(
                    client_id=str(item["client_id"]),
                    host=str(item["host"]),
                    username=str(item.get("username", "jetson")),
                    password=str(item.get("password", "jetson")),
                    remote_root=str(item.get("remote_root", "/home/jetson/L-shaped")),
                )
            )
    return android_specs, jetson_specs


def build_run_id(run_label: str, explicit: str) -> str:
    if explicit:
        return explicit
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_label}"


def phone_remote_power_paths(spec: AndroidSpec, run_id: str) -> dict[str, str]:
    base = posixpath.join(spec.device_root, "outputs", "runs", run_id, spec.client_id)
    return {
        "base": base,
        "csv": posixpath.join(base, "power_samples.csv"),
    }


def jetson_remote_paths(spec: JetsonSpec, run_id: str) -> dict[str, str]:
    base = posixpath.join(spec.remote_root, "outputs", "runs", run_id, spec.client_id)
    return {
        "base": base,
        "power_csv": posixpath.join(base, "power_samples.csv"),
        "power_pid": posixpath.join(base, "power_sampler.pid"),
        "power_stop": posixpath.join(base, "power_sampler.stop"),
    }


def start_phone_power_sampler(
    adb_bin: str,
    spec: AndroidSpec,
    run_id: str,
    run_dir: Path,
    *,
    keep_awake: bool,
) -> dict[str, str]:
    local_client_dir = run_dir / "clients" / spec.client_id
    local_client_dir.mkdir(parents=True, exist_ok=True)
    local_csv = local_client_dir / "power_samples.csv"
    # Always reset first. A previous interrupted run can leave BatteryService in
    # "UPDATES STOPPED" fake-unplug mode, which can trigger Huawei WiFi sleep.
    adb_shell(adb_bin, spec.serial, "dumpsys battery reset >/dev/null 2>&1 || true", timeout=120, check=False)
    if keep_awake:
        adb_shell(
            adb_bin,
            spec.serial,
            "settings put global stay_on_while_plugged_in 7 >/dev/null 2>&1 || true; "
            "settings put global wifi_sleep_policy 2 >/dev/null 2>&1 || true; "
            "cmd deviceidle disable >/dev/null 2>&1 || true; "
            "input keyevent 224 >/dev/null 2>&1 || true",
            timeout=120,
            check=False,
        )
    battery_state = adb_shell(adb_bin, spec.serial, "dumpsys battery", timeout=120, check=False)
    adb_shell(adb_bin, spec.serial, "dumpsys batterystats --reset >/dev/null 2>&1 || true", timeout=120, check=False)
    # Huawei devices often stay fully charged while connected for ADB, which
    # makes batterystats report zero drain. Force a logical unplug for the run
    # and reset it in stop_phone_power_sampler().
    adb_shell(adb_bin, spec.serial, "dumpsys battery unplug >/dev/null 2>&1 || true", timeout=120, check=False)
    return {
        "local_csv": str(local_csv),
        "remote_csv": phone_remote_power_paths(spec, run_id)["csv"],
        "report_path": str(local_client_dir / "batterystats_report.txt"),
        "start_ts": str(time.time()),
        "start_voltage_v": str(parse_battery_voltage_v(battery_state)),
    }


def cleanup_phone_training_processes(adb_bin: str, spec: AndroidSpec) -> None:
    adb_shell(
        adb_bin,
        spec.serial,
        "killall lshaped_flower_client >/dev/null 2>&1 || pkill -f lshaped_flower_client >/dev/null 2>&1 || true",
        timeout=120,
        check=False,
    )


def stop_phone_power_sampler(adb_bin: str, spec: AndroidSpec, sampler: dict[str, str]) -> None:
    start_ts = float(sampler.get("start_ts", "0") or 0.0)
    elapsed_sec = max(0.0, time.time() - start_ts)
    target_window_sec = MIN_RELIABLE_PHONE_POWER_WINDOW_SEC + POWER_WINDOW_EXTRA_SEC
    if 0.0 < elapsed_sec < target_window_sec:
        time.sleep(target_window_sec - elapsed_sec)
    end_ts = time.time()
    batterystats = adb_shell(adb_bin, spec.serial, "dumpsys batterystats", timeout=60, check=False)
    batterystats_checkin = adb_shell(adb_bin, spec.serial, "dumpsys batterystats --checkin", timeout=120, check=False)
    battery_state = adb_shell(adb_bin, spec.serial, "dumpsys battery", timeout=120, check=False)
    adb_shell(
        adb_bin,
        spec.serial,
        "dumpsys battery reset >/dev/null 2>&1 || true; "
        "cmd deviceidle enable >/dev/null 2>&1 || true",
        timeout=120,
        check=False,
    )

    duration_sec = max(0.0, end_ts - start_ts)
    start_voltage_v = float(sampler.get("start_voltage_v", "0") or 0.0)
    end_voltage_v = parse_battery_voltage_v(battery_state)
    voltage_values = [value for value in (start_voltage_v, end_voltage_v) if value > 0.0]
    avg_voltage_v = statistics.fmean(voltage_values) if voltage_values else 0.0
    shell_uid_mah = parse_batterystats_uid_mah(batterystats, 2000)
    computed_drain_mah = parse_batterystats_computed_drain_mah(batterystats)
    battery_capacity_mah = parse_batterystats_capacity_mah(batterystats)
    checkin_capacity_mah, checkin_total_mah, checkin_shell_uid_mah = parse_batterystats_checkin_power_mah(
        batterystats_checkin,
        uid=2000,
    )
    if battery_capacity_mah <= 10.0 and checkin_capacity_mah > 0.0:
        battery_capacity_mah = checkin_capacity_mah
    power_source, power_source_mah, avg_power_w, power_quality, power_flags = select_phone_power_candidate(
        {
            "batterystats_checkin_shell_uid": checkin_shell_uid_mah,
            "batterystats_shell_uid": shell_uid_mah,
            "batterystats_checkin_total": checkin_total_mah,
            "batterystats_computed_drain": computed_drain_mah,
        },
        duration_sec,
        avg_voltage_v,
    )

    report_path = Path(sampler["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n\n".join(
            [
                "=== dumpsys battery ===",
                battery_state,
                "=== dumpsys batterystats ===",
                batterystats,
                "=== dumpsys batterystats --checkin ===",
                batterystats_checkin,
            ]
        ),
        encoding="utf-8",
    )
    write_csv(
        Path(sampler["local_csv"]),
        [
            {
                "start_ts": round(start_ts, 6),
                "end_ts": round(end_ts, 6),
                "duration_sec": round(duration_sec, 6),
                "start_voltage_v": round(start_voltage_v, 6),
                "end_voltage_v": round(end_voltage_v, 6),
                "avg_voltage_v": round(avg_voltage_v, 6),
                "battery_capacity_mah": round(battery_capacity_mah, 6),
                "shell_uid_mah": round(shell_uid_mah, 6),
                "computed_drain_mah": round(computed_drain_mah, 6),
                "checkin_total_mah": round(checkin_total_mah, 6),
                "checkin_shell_uid_mah": round(checkin_shell_uid_mah, 6),
                "power_source": power_source,
                "power_source_mah": round(power_source_mah, 6),
                "power_quality": power_quality,
                "power_flags": ";".join(power_flags),
                "avg_power_w": round(avg_power_w, 6),
            }
        ],
    )


def select_phone_power_candidate(
    candidates: dict[str, float],
    duration_sec: float,
    avg_voltage_v: float,
) -> tuple[str, float, float, str, list[str]]:
    rejected_sources: list[str] = []
    if duration_sec <= 0.0 or avg_voltage_v <= 0.0:
        return "", 0.0, 0.0, "missing", ["missing_duration_or_voltage"]
    short_window = duration_sec < (MIN_RELIABLE_PHONE_POWER_WINDOW_SEC - POWER_WINDOW_TOLERANCE_SEC)

    def candidate_power(source: str) -> tuple[str, float, float]:
        mah = candidates.get(source, 0.0)
        if mah <= 0.0:
            return source, mah, 0.0
        return source, mah, (mah / 1000.0) * avg_voltage_v / (duration_sec / 3600.0)

    for source in ("batterystats_checkin_shell_uid", "batterystats_shell_uid"):
        source, mah, avg_power_w = candidate_power(source)
        if mah <= 0.0:
            continue
        flags: list[str] = []
        if short_window:
            flags.append("short_window_lt300s")
        if avg_power_w > MAX_PHONE_UID_POWER_W:
            rejected_sources.append(source)
            continue
        quality = "warn" if flags else "ok"
        return source, mah, avg_power_w, quality, flags

    for source in ("batterystats_checkin_total", "batterystats_computed_drain"):
        source, mah, avg_power_w = candidate_power(source)
        if mah <= 0.0:
            continue
        flags = ["fallback_device_level_power"]
        if short_window:
            flags.append("short_window_lt300s")
        if avg_power_w > MAX_PHONE_AVG_POWER_W:
            rejected_sources.append(source)
            continue
        quality = "warn" if flags else "ok"
        return source, mah, avg_power_w, quality, flags

    best_source = ""
    best_mah = 0.0
    best_power = 0.0
    for source in (
        "batterystats_checkin_shell_uid",
        "batterystats_shell_uid",
        "batterystats_checkin_total",
        "batterystats_computed_drain",
    ):
        source, mah, avg_power_w = candidate_power(source)
        if avg_power_w > best_power:
            best_source, best_mah, best_power = source, mah, avg_power_w
    if best_power > 0.0:
        flags = ["implausible_power_retained_for_audit"]
        if short_window:
            flags.append("short_window_lt300s")
        if rejected_sources:
            flags.append("rejected_sources=" + "+".join(rejected_sources))
        return best_source, best_mah, best_power, "questionable", flags
    return "", 0.0, 0.0, "missing", ["no_positive_power_candidate"]


def start_jetson_power_sampler(spec: JetsonSpec, run_id: str, run_dir: Path) -> dict[str, str]:
    paths = jetson_remote_paths(spec, run_id)
    local_client_dir = run_dir / "clients" / spec.client_id
    local_client_dir.mkdir(parents=True, exist_ok=True)
    local_csv = local_client_dir / "power_samples.csv"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        client = connect_password(spec.host, spec.username, spec.password)
        try:
            code, out, _ = run_remote(client, "systemctl is-active jtop.service || true", get_pty=False)
            if code != 0 or out.strip() != "active":
                code, out, err = run_remote_sudo(client, spec.password, "systemctl restart jtop.service")
                if code != 0:
                    raise RuntimeError(
                        f"Failed to restart jtop.service on {spec.host}\nstdout:\n{out}\nstderr:\n{err}"
                    )
                time.sleep(2)
            sampler_code = textwrap.dedent(
                f'''
import csv
import os
import pathlib
import time
from jtop import jtop

csv_path = {paths['power_csv']!r}
stop_path = {paths['power_stop']!r}
pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
with open(csv_path, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["ts", "power_mw"])
    writer.writeheader()
    with jtop() as jetson:
        while jetson.ok():
            if os.path.exists(stop_path):
                break
            stats = jetson.stats
            writer.writerow({{"ts": time.time(), "power_mw": stats.get("Power TOT", 0)}})
            handle.flush()
            time.sleep(1.0)
'''
            ).strip()
            script = textwrap.dedent(
                f"""
import pathlib
import subprocess

csv_path = {paths['power_csv']!r}
stop_path = {paths['power_stop']!r}
pid_path = {paths['power_pid']!r}
sampler = {sampler_code!r}
pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
try:
    pathlib.Path(stop_path).unlink()
except FileNotFoundError:
    pass
proc = subprocess.Popen(
    ["python3", "-c", sampler],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
pathlib.Path(pid_path).write_text(str(proc.pid), encoding="utf-8")
print(proc.pid)
"""
            ).strip()
            code, out, err = run_remote(client, f"python3 -c {shlex.quote(script)}", get_pty=False)
            if code != 0:
                raise RuntimeError(f"Failed to start Jetson power sampler on {spec.host}\nstdout:\n{out}\nstderr:\n{err}")
            pid = out.strip().splitlines()[-1].strip()
            ready = False
            for _ in range(10):
                code, out, _ = run_remote(
                    client,
                    textwrap.dedent(
                        f"""
                        if [ -f {shlex.quote(paths['power_csv'])} ]; then
                          wc -l < {shlex.quote(paths['power_csv'])}
                        else
                          echo 0
                        fi
                        """
                    ).strip(),
                    get_pty=False,
                )
                if code == 0 and int((out.strip() or "0")) >= 2:
                    ready = True
                    break
                time.sleep(1)
            if ready:
                return {"pid": pid, "local_csv": str(local_csv), **paths}
            run_remote(
                client,
                f"touch {shlex.quote(paths['power_stop'])}; "
                f"if [ -f {shlex.quote(paths['power_pid'])} ]; then kill $(cat {shlex.quote(paths['power_pid'])}) >/dev/null 2>&1 || true; fi",
                get_pty=False,
            )
            raise RuntimeError(f"Jetson power sampler on {spec.host} did not emit any samples")
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2)
        finally:
            client.close()
    assert last_error is not None
    raise last_error


def stop_jetson_power_sampler(spec: JetsonSpec, sampler: dict[str, str]) -> None:
    try:
        client = connect_password(spec.host, spec.username, spec.password)
    except Exception as exc:
        log(f"[split-boolq] warning: failed to reconnect to {spec.host} to stop power sampler: {exc}")
        return
    try:
        run_remote(
            client,
            f"touch {shlex.quote(sampler['power_stop'])}; "
            f"if [ -f {shlex.quote(sampler['power_pid'])} ]; then kill $(cat {shlex.quote(sampler['power_pid'])}) >/dev/null 2>&1 || true; fi",
            get_pty=False,
        )
        try:
            sftp = client.open_sftp()
            try:
                local_csv = Path(sampler["local_csv"])
                local_csv.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(sampler["power_csv"], str(local_csv))
            finally:
                sftp.close()
        except FileNotFoundError:
            pass
        except Exception as exc:
            log(f"[split-boolq] warning: failed to pull power CSV from {spec.host}: {exc}")
    except Exception as exc:
        log(f"[split-boolq] warning: failed to stop Jetson power sampler on {spec.host}: {exc}")
    finally:
        client.close()


def cleanup_jetson_training_processes(spec: JetsonSpec) -> None:
    client = connect_password(spec.host, spec.username, spec.password)
    try:
        run_remote(
            client,
            "for pid in $(pgrep -f '[l]shaped_flower_client' 2>/dev/null || true); do "
            "  kill \"$pid\" >/dev/null 2>&1 || true; "
            "done",
            get_pty=False,
        )
    finally:
        client.close()


def parse_power_csv(path: Path, *, kind: str) -> tuple[float, int]:
    if not path.is_file():
        return (0.0, 0)
    samples: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if kind == "jetson":
                raw = row.get("power_mw", "").strip()
                if raw:
                    samples.append(float(raw) / 1000.0)
            else:
                avg_power = row.get("avg_power_w", "").strip()
                if avg_power:
                    samples.append(float(avg_power))
    if not samples:
        return (0.0, 0)
    return (statistics.fmean(samples), len(samples))


def parse_battery_voltage_v(text: str) -> float:
    match = re.search(r"^\s*voltage:\s*([0-9]+)\s*$", text, flags=re.MULTILINE)
    if not match:
        return 0.0
    return float(match.group(1)) / 1000.0


def parse_batterystats_uid_mah(text: str, uid: int) -> float:
    match = re.search(rf"Uid\s+{uid}:\s+([0-9.]+)", text)
    if not match:
        return 0.0
    return float(match.group(1))


def parse_batterystats_computed_drain_mah(text: str) -> float:
    match = re.search(r"Computed drain:\s*([0-9.]+)", text)
    if not match:
        return 0.0
    return float(match.group(1))


def parse_batterystats_capacity_mah(text: str) -> float:
    match = re.search(r"Capacity:\s*([0-9.]+)", text)
    if not match:
        return 0.0
    return float(match.group(1))


def parse_batterystats_checkin_power_mah(text: str, *, uid: int) -> tuple[float, float, float]:
    capacity_mah = 0.0
    total_mah = 0.0
    uid_mah = 0.0
    for row in csv.reader(text.splitlines()):
        if len(row) >= 6 and row[3] == "pws":
            capacity_mah = max(capacity_mah, to_float(row[4]))
            total_mah = max(total_mah, to_float(row[5]))
        elif len(row) >= 6 and row[3] == "pwi" and row[1] == str(uid) and row[4] == "uid":
            uid_mah = max(uid_mah, to_float(row[5]))
    return capacity_mah, total_mah, uid_mah


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_float(raw: Any, default: float = 0.0) -> float:
    if raw in ("", None):
        return default
    return float(raw)


def to_int(raw: Any, default: int = 0) -> int:
    if raw in ("", None):
        return default
    return int(float(raw))


def main() -> None:
    args = parse_args()
    root = repo_root()
    adb_bin = resolve_adb_path(args.adb_path)
    run_id = build_run_id(args.run_label, args.run_id)
    run_dir = root / "L-shaped_code_docs_backup" / "legacy_split" / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    android_specs, jetson_specs = load_specs(resolve_local_path(root, args.client_specs_json))

    base_config_path = resolve_local_path(root, args.base_config)
    cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not args.skip_prepare_script:
        prepare_script = args.prepare_dataset_script or args.prepare_boolq_script
        log(f"[split-measurement] preparing local dataset with {prepare_script}")
        prepare_cmd = [
            "python3",
            str(resolve_local_path(root, prepare_script)),
            *args.prepare_script_extra_args,
        ]
        if args.prepare_script_output_dir:
            prepare_cmd.extend(["--output-dir", args.prepare_script_output_dir])
        if args.prepare_script_model_dir:
            prepare_cmd.extend(["--model-dir", args.prepare_script_model_dir])
        if args.prepare_script_seq_len > 0:
            prepare_cmd.extend(["--seq-len", str(args.prepare_script_seq_len)])
        run_local(
            prepare_cmd,
            cwd=root,
            timeout=7200,
        )
    else:
        log("[split-measurement] skipping dataset preparation; using dataset.source_path")
    dataset_source_path = str(cfg.get("dataset", {}).get("source_path", "")).strip()
    if not dataset_source_path:
        raise RuntimeError(f"Missing dataset.source_path in config: {base_config_path}")
    train_csv_path = resolve_local_path(root, dataset_source_path)
    assert train_csv_path.is_file(), f"Missing prepared train CSV: {train_csv_path}"

    phone_samplers: dict[str, dict[str, str]] = {}
    jetson_samplers: dict[str, dict[str, str]] = {}
    try:
        log("[split-measurement] cleaning stale phone training processes")
        for spec in android_specs:
            cleanup_phone_training_processes(adb_bin, spec)
        log("[split-measurement] cleaning stale jetson training processes")
        for spec in jetson_specs:
            cleanup_jetson_training_processes(spec)
        log("[split-measurement] starting phone power samplers")
        for spec in android_specs:
            phone_samplers[spec.client_id] = start_phone_power_sampler(
                adb_bin,
                spec,
                run_id,
                run_dir,
                keep_awake=not args.no_phone_keep_awake,
            )
        log("[split-measurement] starting jetson power samplers")
        for spec in jetson_specs:
            jetson_samplers[spec.client_id] = start_jetson_power_sampler(spec, run_id, run_dir)

        launch_cmd = [
            "python3",
            str(root / "L-shaped_code_docs_backup" / "legacy_split" / "scripts" / "run_mixed_client_experiment.py"),
            "--base-config",
            str(resolve_local_path(root, args.base_config)),
            "--client-specs-json",
            str(resolve_local_path(root, args.client_specs_json)),
            "--run-id",
            run_id,
            "--run-label",
            args.run_label,
            "--shared-client-dataset-local-csv",
            str(train_csv_path),
            "--server-address-host",
            args.server_address_host,
            "--server-ssh-host",
            args.server_ssh_host,
            "--server-ssh-username",
            args.server_ssh_username,
            "--server-remote-root",
            args.server_remote_root,
            "--server-python",
            args.server_python,
            "--cuda-visible-devices",
            args.cuda_visible_devices,
            "--server-wait-timeout",
            str(args.server_wait_timeout),
            "--server-exit-timeout",
            str(args.server_exit_timeout),
            "--client-exit-timeout",
            str(args.client_exit_timeout),
            "--connect-max-attempts",
            str(args.connect_max_attempts),
            "--connect-ready-timeout-ms",
            str(args.connect_ready_timeout_ms),
            "--connect-retry-delay-ms",
            str(args.connect_retry_delay_ms),
            "--default-nano-password",
            args.default_nano_password,
            "--default-nano-remote-root",
            args.default_nano_remote_root,
            "--default-android-stage-local-dir",
            str(root / "L-shaped_code_docs_backup" / "outputs" / "android_client" / "arm64-v8a" / "mft"),
            "--adb-path",
            adb_bin,
        ]
        if args.skip_android_binary_push:
            launch_cmd.append("--skip-android-binary-push")
        if args.skip_android_model_push:
            launch_cmd.append("--skip-android-model-push")

        log("[split-measurement] launching mixed SplitLoRA run")
        run_local(launch_cmd, cwd=root, timeout=args.timeout_sec)
    finally:
        log("[split-measurement] stopping power samplers")
        for spec in android_specs:
            sampler = phone_samplers.get(spec.client_id)
            if sampler:
                stop_phone_power_sampler(adb_bin, spec, sampler)
        for spec in jetson_specs:
            sampler = jetson_samplers.get(spec.client_id)
            if sampler:
                stop_jetson_power_sampler(spec, sampler)

    server_dir = run_dir / "server"
    metrics_path = server_dir / "metrics.csv"
    summary_path = server_dir / "summary_rounds.csv"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"Missing split run outputs under {server_dir}")

    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("mode") == "train" and row.get("round") == "1":
                rows.append(dict(row))

    power_rows: list[dict[str, Any]] = []
    power_index: dict[str, dict[str, Any]] = {}
    for spec in android_specs:
        local_path = run_dir / "clients" / spec.client_id / "power_samples.csv"
        mean_power_w, num_samples = parse_power_csv(local_path, kind="phone")
        row = {"client_id": spec.client_id, "avg_power_w": round(mean_power_w, 6), "power_samples": num_samples}
        power_rows.append(row)
        power_index[spec.client_id] = row
    for spec in jetson_specs:
        local_path = run_dir / "clients" / spec.client_id / "power_samples.csv"
        mean_power_w, num_samples = parse_power_csv(local_path, kind="jetson")
        row = {"client_id": spec.client_id, "avg_power_w": round(mean_power_w, 6), "power_samples": num_samples}
        power_rows.append(row)
        power_index[spec.client_id] = row

    local_client_metrics: dict[str, dict[str, Any]] = {}
    for spec in android_specs:
        local_metrics_path = run_dir / "clients" / spec.client_id / "client_metrics.csv"
        if not local_metrics_path.is_file():
            continue
        with local_metrics_path.open("r", encoding="utf-8") as handle:
            for local_row in csv.DictReader(handle):
                if str(local_row.get("server_round", "")) == "1":
                    local_client_metrics[spec.client_id] = dict(local_row)
    for spec in jetson_specs:
        local_metrics_path = run_dir / "clients" / spec.client_id / "client_metrics.csv"
        if not local_metrics_path.is_file():
            continue
        with local_metrics_path.open("r", encoding="utf-8") as handle:
            for local_row in csv.DictReader(handle):
                if str(local_row.get("server_round", "")) == "1":
                    local_client_metrics[spec.client_id] = dict(local_row)

    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        local_metric = local_client_metrics.get(row["client_id"], {})
        local_download_time_sec = to_float(local_metric.get("download_time_sec"))
        local_upload_time_sec = to_float(local_metric.get("upload_time_sec"))
        step_times_raw = row.get("step_times_sec_json", "")
        step_times: list[float] = []
        if step_times_raw:
            try:
                parsed = json.loads(step_times_raw)
                if isinstance(parsed, list):
                    step_times = [float(item) for item in parsed]
            except json.JSONDecodeError:
                step_times = []
        steps_completed = len(step_times) if step_times else to_int(row.get("steps_completed"))
        mean_step_time_sec = statistics.fmean(step_times) if step_times else to_float(row.get("mean_step_time_sec"))
        max_step_time_sec = max(step_times) if step_times else to_float(row.get("max_step_time_sec"))
        merged = {
            "client_id": row["client_id"],
            "steps_completed": steps_completed,
            "num_examples": to_int(row.get("total_examples", row.get("num_examples"))),
            "loss": to_float(row.get("loss")),
            "accuracy": to_float(row.get("accuracy")),
            "client_encode_time_sec": to_float(row.get("client_encode_time_sec")),
            "client_serialize_time_sec": to_float(row.get("client_serialize_time_sec")),
            "client_round_time_sec": to_float(row.get("client_round_time_sec")),
            "mean_step_time_sec": mean_step_time_sec,
            "max_step_time_sec": max_step_time_sec,
            "step_times_sec_json": step_times_raw,
            "download_time_sec": (
                local_download_time_sec if local_download_time_sec > 0.0 else to_float(row.get("download_time_sec"))
            ),
            "upload_time_sec": (
                local_upload_time_sec if local_upload_time_sec > 0.0 else to_float(row.get("upload_time_sec"))
            ),
            "download_bytes": to_int(row.get("download_bytes")),
            "upload_bytes": to_int(row.get("upload_bytes")),
            "transmitted_bytes": to_int(row.get("transmitted_bytes")),
            "avg_rss_mb": to_float(row.get("avg_rss_mb"), -1.0),
            "peak_rss_mb": to_float(row.get("peak_rss_mb"), -1.0),
            "server_rss_mb": to_float(row.get("rss_mb"), -1.0),
            "avg_power_w": power_index.get(row["client_id"], {}).get("avg_power_w", 0.0),
            "power_samples": power_index.get(row["client_id"], {}).get("power_samples", 0),
        }
        clean_rows.append(merged)

    with summary_path.open("r", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    if not summary_rows:
        raise RuntimeError(f"Empty split summary: {summary_path}")
    round1 = summary_rows[0]
    clean_summary_row = {
        "round": round1.get("round", "1"),
        "num_clients": len(clean_rows),
        "num_failures": to_int(round1.get("num_failures")),
        "total_examples": sum(to_int(item.get("num_examples")) for item in clean_rows),
        "mean_loss": statistics.fmean(to_float(item.get("loss")) for item in clean_rows) if clean_rows else 0.0,
        "mean_accuracy": statistics.fmean(to_float(item.get("accuracy")) for item in clean_rows) if clean_rows else 0.0,
        "mean_step_time_sec": statistics.fmean(to_float(item.get("mean_step_time_sec")) for item in clean_rows) if clean_rows else 0.0,
        "max_step_time_sec": max((to_float(item.get("max_step_time_sec")) for item in clean_rows), default=0.0),
        "aggregation_time_sec": to_float(round1.get("aggregation_time_sec")),
        "mean_download_time_sec": statistics.fmean(to_float(item.get("download_time_sec")) for item in clean_rows) if clean_rows else 0.0,
        "mean_upload_time_sec": statistics.fmean(to_float(item.get("upload_time_sec")) for item in clean_rows) if clean_rows else 0.0,
        "mean_download_bytes": statistics.fmean(to_int(item.get("download_bytes")) for item in clean_rows) if clean_rows else 0.0,
        "mean_upload_bytes": statistics.fmean(to_int(item.get("upload_bytes")) for item in clean_rows) if clean_rows else 0.0,
        "mean_transmitted_bytes": statistics.fmean(to_int(item.get("transmitted_bytes")) for item in clean_rows) if clean_rows else 0.0,
        "mean_avg_rss_mb": statistics.fmean(to_float(item.get("avg_rss_mb")) for item in clean_rows) if clean_rows else 0.0,
        "mean_peak_rss_mb": statistics.fmean(to_float(item.get("peak_rss_mb")) for item in clean_rows) if clean_rows else 0.0,
        "mean_client_power_w": statistics.fmean(to_float(item.get("avg_power_w")) for item in clean_rows) if clean_rows else 0.0,
    }

    write_csv(server_dir / "power_summary.csv", power_rows)
    write_csv(server_dir / "round1_client_summary_clean.csv", clean_rows)
    write_csv(server_dir / "summary_rounds_clean.csv", [clean_summary_row])
    summary_payload = {
        "run_id": run_id,
        "round1": clean_summary_row,
        "client_count": len(clean_rows),
    }
    (server_dir / "round1_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run_boolq_split_measurement] {exc}", file=sys.stderr)
        raise
