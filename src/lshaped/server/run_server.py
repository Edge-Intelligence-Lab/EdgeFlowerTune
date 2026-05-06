from __future__ import annotations

import argparse
import random

import numpy as np
from flwr.server import ServerConfig, start_server

from lshaped.common.logging_utils import configure_logging
from lshaped.config import load_config
from lshaped.server.classic_fedavg_strategy import ClassicFedAvgLoraStrategy
from lshaped.server.flexlora_strategy import ClassicFlexLoraStrategy
from lshaped.server.local_only_strategy import LocalOnlyLoraStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classic FL + LoRA Flower server")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def relax_grpc_keepalive_policy() -> None:
    """Allow slow edge clients to keep long-running Flower streams alive.

    Jetson clients can spend ~2 minutes inside one local Gemma step with no
    application-level messages on the bidirectional stream. Client keepalive
    avoids HTTP/2 settings timeouts on that path, so the server must accept
    those pings instead of closing the connection with ENHANCE_YOUR_CALM.
    """
    import concurrent.futures

    import grpc
    import flwr.server.superlink.fleet.grpc_bidi.grpc_server as grpc_server_module
    from flwr.common.grpc import is_port_in_use, valid_certificates

    def create_server_with_relaxed_keepalive(
        servicer_and_add_fn,
        server_address: str,
        max_concurrent_workers: int = 1000,
        max_message_length: int = 536870912,
        keepalive_time_ms: int = 210000,
        certificates=None,
        interceptors=None,
    ):
        if is_port_in_use(server_address):
            raise SystemExit(f"Port in server address {server_address} is already in use.")

        servicer, add_servicer_to_server_fn = servicer_and_add_fn
        options = [
            ("grpc.max_concurrent_streams", max(100, max_concurrent_workers)),
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
            ("grpc.keepalive_time_ms", keepalive_time_ms),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.keepalive_permit_without_calls", 0),
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
            ("grpc.http2.max_ping_strikes", 0),
        ]
        server = grpc.server(
            concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_workers),
            maximum_concurrent_rpcs=max_concurrent_workers,
            options=options,
            interceptors=interceptors,
        )
        add_servicer_to_server_fn(servicer, server)

        if certificates is not None:
            if not valid_certificates(certificates):
                raise SystemExit(1)
            root_certificate_b, certificate_b, private_key_b = certificates
            server_credentials = grpc.ssl_server_credentials(
                ((private_key_b, certificate_b),),
                root_certificates=root_certificate_b,
                require_client_auth=False,
            )
            server.add_secure_port(server_address, server_credentials)
        else:
            server.add_insecure_port(server_address)

        return server

    grpc_server_module.generic_create_grpc_server = create_server_with_relaxed_keepalive


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.runtime.seed)
    logger = configure_logging(cfg.runtime.output_dir, "server")
    relax_grpc_keepalive_policy()
    if cfg.federated.algorithm == "splitlora":
        from lshaped.server.split_strategy import SplitLoraStrategy

        strategy = SplitLoraStrategy(cfg=cfg, logger=logger)
    elif cfg.federated.algorithm == "localonly_lora":
        strategy = LocalOnlyLoraStrategy(cfg=cfg, logger=logger)
    elif cfg.federated.algorithm == "flexlora":
        strategy = ClassicFlexLoraStrategy(cfg=cfg, logger=logger)
    else:
        strategy = ClassicFedAvgLoraStrategy(cfg=cfg, logger=logger)

    logger.info(
        "Starting classic %s server on %s",
        cfg.federated.algorithm,
        cfg.flower.server_address,
    )
    start_server(
        server_address=cfg.flower.server_address,
        config=ServerConfig(num_rounds=cfg.flower.num_rounds, round_timeout=cfg.flower.round_timeout),
        strategy=strategy,
        grpc_max_message_length=cfg.flower.grpc_max_message_length,
    )


if __name__ == "__main__":
    main()
