from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run split WikiText checkpoint eval on server3 with multiple GPUs")
    parser.add_argument("--host", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--remote-eval-dir", required=True)
    parser.add_argument("--remote-checkpoint-dir", required=True)
    parser.add_argument("--remote-eval-raw", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--local-eval-script", required=True)
    parser.add_argument("--local-common-lora", required=True)
    return parser.parse_args()


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = [part for part in remote_dir.strip("/").split("/") if part]
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_file(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    sftp_mkdirs(sftp, str(Path(remote_path).parent).replace("\\", "/"))
    sftp.put(str(local_path), remote_path)


def main() -> None:
    args = parse_args()

    eval_script = Path(args.local_eval_script).resolve()
    common_lora = Path(args.local_common_lora).resolve()
    if not eval_script.is_file():
        raise FileNotFoundError(eval_script)
    if not common_lora.is_file():
        raise FileNotFoundError(common_lora)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.username, timeout=20)
    sftp = client.open_sftp()
    try:
        sftp_mkdirs(sftp, args.remote_eval_dir)
        upload_file(sftp, eval_script, f"{args.remote_eval_dir}/eval_mmlu_checkpoints.py")
        upload_file(sftp, common_lora, f"{args.remote_eval_dir}/common_lora.py")

        script = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            cd {args.remote_eval_dir}
            PY={args.python_bin}
            MODEL={args.model_dir}
            CKPT={args.remote_checkpoint_dir}
            RAW={args.remote_eval_raw}
            COMMON_ARGS=(eval_mmlu_checkpoints.py --model-dir "$MODEL" --checkpoint-dir "$CKPT" --dataset-format wikitext_raw --eval-raw "$RAW" --seq-len 64 --lora-r 8 --lora-alpha 16 --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)
            CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "${{COMMON_ARGS[@]}}" --batch-size 16 --only-rounds 1-25 --output-csv wikitext_curve_part1.csv --output-json wikitext_curve_part1.json > wikitext_curve_part1.log 2>&1 &
            P1=$!
            CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "${{COMMON_ARGS[@]}}" --batch-size 16 --only-rounds 26-50 --output-csv wikitext_curve_part2.csv --output-json wikitext_curve_part2.json > wikitext_curve_part2.log 2>&1 &
            P2=$!
            CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "${{COMMON_ARGS[@]}}" --batch-size 16 --only-rounds 51-75 --output-csv wikitext_curve_part3.csv --output-json wikitext_curve_part3.json > wikitext_curve_part3.log 2>&1 &
            P3=$!
            CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "${{COMMON_ARGS[@]}}" --batch-size 16 --only-rounds 76-100 --output-csv wikitext_curve_part4.csv --output-json wikitext_curve_part4.json > wikitext_curve_part4.log 2>&1 &
            P4=$!
            wait "$P1" "$P2" "$P3" "$P4"
            python3 - <<'PY'
            import csv, json
            from pathlib import Path

            base = Path("{args.remote_eval_dir}")
            rows = []
            for idx in range(1, 5):
                part = base / f"wikitext_curve_part{{idx}}.csv"
                with part.open() as handle:
                    rows.extend(list(csv.DictReader(handle)))
            rows.sort(key=lambda row: int(row["round"]))
            out_csv = base / "wikitext_test_curve.csv"
            out_json = base / "wikitext_test_curve.json"
            with out_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "round",
                        "checkpoint",
                        "eval_loss",
                        "eval_perplexity",
                        "eval_accuracy",
                        "eval_examples",
                        "eval_tokens",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
            out_json.write_text(
                json.dumps({{"num_checkpoints": len(rows), "rows": rows}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(out_csv)
            print(out_json)
            PY
            """
        )

        local_script = Path("/tmp/run_split_wikitext_curve_full.sh")
        local_script.write_text(script, encoding="utf-8")
        remote_script = f"{args.remote_eval_dir}/run_full_curve.sh"
        upload_file(sftp, local_script, remote_script)
        sftp.chmod(remote_script, 0o755)
    finally:
        sftp.close()
        client.close()

    print(f"{args.remote_eval_dir}/run_full_curve.sh")


if __name__ == "__main__":
    main()
