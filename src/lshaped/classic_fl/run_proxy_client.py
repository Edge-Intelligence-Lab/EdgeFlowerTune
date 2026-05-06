from __future__ import annotations

from lshaped.classic_fl.device_proxy_client import load_spec, parse_args, run_client


def main() -> None:
    args = parse_args()
    spec = load_spec(args.spec_json)
    run_client(spec, server_address=args.server_address, grpc_max_message_length=args.grpc_max_message_length)


if __name__ == "__main__":
    main()
