# Centralized LoRA Architecture

This baseline is the higher-bound reference.

## Topology

- `server3` trains Gemma-270M with LoRA on the union of the client data
- the training data is the same global CSV whose round-robin partitions were used by the `3` nova clients
- no federated aggregation is involved
- Flower is not part of the optimization loop for this baseline

## Training Flow

1. load the full `official_mmlu_test_100.csv`
2. convert each row into the same masked causal-LM format used by the edge LoRA clients
3. train one server-side LoRA adapter on the merged dataset
4. save step metrics and adapter checkpoints

## Why This Is The Higher Bound

- all training samples are visible to a single optimizer state
- there is no statistical heterogeneity across clients during optimization
- there is no communication loss or aggregation noise

So this is the natural upper reference against the federated baselines.

## Checkpoints And Logs

- step metrics are written to `metrics.csv`
- summary information is written to `summary.json`
- LoRA checkpoints are written under `checkpoints/`

## References

- FedAvg: [PMLR 54, 2017](https://proceedings.mlr.press/v54/mcmahan17a.html)
- LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
