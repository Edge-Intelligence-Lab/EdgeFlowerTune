from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one L-shaped run directory into compact CSV/JSON tables")
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def _to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _to_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def _mean(values: list[float]) -> float:
    assert values, "mean requires non-empty list"
    return sum(values) / len(values)


def _mode(row: dict[str, str]) -> str:
    return row.get("mode") or "train"


def _first_available_float(row: dict[str, str], keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if row.get(key, "") != "":
            return _to_float(row, key, default)
    return default


def summarize_server_metrics(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "server" / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing server metrics CSV: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"Server metrics CSV is empty: {metrics_path}"

    by_round: dict[int, list[dict[str, str]]] = defaultdict(list)
    by_client_mode: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        round_id = _to_int(row, "round")
        by_round[round_id].append(row)
        by_client_mode[(row["client_id"], _mode(row))].append(row)

    round_rows: list[dict[str, object]] = []
    for round_id in sorted(by_round):
        group = by_round[round_id]
        losses = [_to_float(r, "loss") for r in group]
        accuracies = [_to_float(r, "accuracy") for r in group]
        client_times = [
            _first_available_float(r, ["client_round_time_sec", "train_time_sec"])
            for r in group
        ]
        server_times = [_to_float(r, "round_time_sec") for r in group]
        bytes_sent = [_to_float(r, "transmitted_bytes") for r in group]
        gpu_mem = [_to_float(r, "gpu_mem_mb") for r in group]
        gpu_power = [_to_float(r, "gpu_power_w") for r in group]
        queue_size = [_to_float(r, "queue_size") for r in group]
        mode = _mode(group[0])
        round_rows.append(
            {
                "round": round_id,
                "mode": mode,
                "num_clients": len(group),
                "mean_loss": _mean(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
                "mean_accuracy": _mean(accuracies),
                "mean_client_round_time_sec": _mean(client_times),
                "mean_server_round_time_sec": _mean(server_times),
                "total_transmitted_bytes": int(sum(bytes_sent)),
                "mean_gpu_mem_mb": _mean(gpu_mem),
                "mean_gpu_power_w": _mean(gpu_power),
                "last_queue_size": int(queue_size[-1]),
            }
        )

    client_rows: list[dict[str, object]] = []
    for (client_id, mode), group in sorted(by_client_mode.items()):
        losses = [_to_float(r, "loss") for r in group]
        accuracies = [_to_float(r, "accuracy") for r in group]
        bytes_sent = [_to_float(r, "transmitted_bytes") for r in group]
        client_rows.append(
            {
                "client_id": client_id,
                "mode": mode,
                "steps": len(group),
                "mean_loss": _mean(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
                "mean_accuracy": _mean(accuracies),
                "total_transmitted_bytes": int(sum(bytes_sent)),
                "mean_client_round_time_sec": _mean([
                    _first_available_float(r, ["client_round_time_sec", "train_time_sec"])
                    for r in group
                ]),
                "mean_server_round_time_sec": _mean([_to_float(r, "round_time_sec") for r in group]),
                "mean_client_rss_mb": _mean([_to_float(r, "client_rss_mb") for r in group]),
                "mean_gpu_mem_mb": _mean([_to_float(r, "gpu_mem_mb") for r in group]),
                "mean_gpu_power_w": _mean([_to_float(r, "gpu_power_w") for r in group]),
            }
        )

    round_csv = run_dir / "summary_rounds.csv"
    with round_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(round_rows[0].keys()))
        writer.writeheader()
        writer.writerows(round_rows)

    client_csv = run_dir / "summary_clients.csv"
    with client_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(client_rows[0].keys()))
        writer.writeheader()
        writer.writerows(client_rows)

    first_train = next((r for r in round_rows if r["mode"] == "train"), None)
    last_train = next((r for r in reversed(round_rows) if r["mode"] == "train"), None)
    first_eval = next((r for r in round_rows if r["mode"] == "eval"), None)
    last_eval = next((r for r in reversed(round_rows) if r["mode"] == "eval"), None)
    summary = {
        "server_metrics_csv": str(metrics_path),
        "summary_rounds_csv": str(round_csv),
        "summary_clients_csv": str(client_csv),
        "num_server_rows": len(rows),
        "num_rounds": len(round_rows),
        "num_clients_observed": len({row["client_id"] for row in rows}),
        "first_train_mean_loss": first_train["mean_loss"] if first_train else None,
        "last_train_mean_loss": last_train["mean_loss"] if last_train else None,
        "first_eval_mean_loss": first_eval["mean_loss"] if first_eval else None,
        "last_eval_mean_loss": last_eval["mean_loss"] if last_eval else None,
        "round_rows": round_rows,
        "client_rows": client_rows,
    }
    return summary


def summarize_client_uploads(run_dir: Path) -> dict[str, object]:
    client_csvs = sorted(run_dir.glob("clients/*/client_metrics.csv"))
    if not client_csvs:
        return {"num_client_metric_files": 0, "clients": []}

    client_rows: list[dict[str, object]] = []
    for path in client_csvs:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        bytes_sent = [_to_float(r, "transmitted_bytes") for r in rows]
        round_time = [_to_float(r, "round_time_sec") for r in rows]
        encode_time = [_to_float(r, "encode_time_sec") for r in rows]
        rss_mb = [_to_float(r, "client_rss_mb") for r in rows]
        client_rows.append(
            {
                "client_id": rows[0]["client_id"],
                "steps": len(rows),
                "total_transmitted_bytes": int(sum(bytes_sent)),
                "mean_round_time_sec": _mean(round_time),
                "mean_encode_time_sec": _mean(encode_time),
                "mean_client_rss_mb": _mean(rss_mb),
            }
        )
    return {
        "num_client_metric_files": len(client_csvs),
        "clients": client_rows,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Missing run directory: {run_dir}")

    server_summary = summarize_server_metrics(run_dir)
    client_summary = summarize_client_uploads(run_dir)
    output = {
        "run_dir": str(run_dir),
        "server": server_summary,
        "client_uploads": client_summary,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"run_dir={run_dir}")
    print(f"summary_json={summary_path}")
    print(f"summary_rounds_csv={run_dir / 'summary_rounds.csv'}")
    print(f"summary_clients_csv={run_dir / 'summary_clients.csv'}")


if __name__ == "__main__":
    main()
