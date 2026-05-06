# Local-only LoRA Architecture

This baseline is the classic lower-bound reference for federated LoRA.

## Topology

- `3` nova phones keep the full Gemma-270M base plus local LoRA adapters
- `server3` still runs Flower and coordinates rounds
- each phone trains only on its own local shard
- `server3` does not aggregate adapters across clients

## Per-Round Flow

1. `server3` keeps one independent LoRA state per client
2. at round start, Flower sends each client only its own previous adapter
3. the client performs local LoRA training on its own shard
4. the client uploads its updated adapter back to `server3`
5. `server3` stores that adapter as the next-round state for that same client
6. no cross-client parameter averaging is performed

## What Makes This Local-only

- communication and orchestration still use Flower
- model updates remain fully personalized
- there is no FedAvg, FedProx, or any other cross-client aggregation
- this is the lower-bound collaborative baseline because every client learns alone

## Checkpoints And Logs

- client metrics are written to `server/metrics.csv`
- per-client adapter checkpoints are written under `server/checkpoints/<client_id>/`
- each client keeps its own LoRA trajectory across rounds

## References

- FedAvg: [PMLR 54, 2017](https://proceedings.mlr.press/v54/mcmahan17a.html)
- LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
