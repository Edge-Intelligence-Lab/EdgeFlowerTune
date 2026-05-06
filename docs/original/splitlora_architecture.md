# SplitLoRA Architecture

This baseline is the split fine-tuning path, not the classic adapter-FL path.

## Topology

- `3` nova phones run the MobileFineTuner Gemma-270M prefix locally
- each phone uploads split activations plus target embeddings to `server3`
- `server3` holds the trainable Gemma suffix with LoRA attached
- Flower is used as the round controller and transport layer

## Per-Round Flow

1. `server3` starts one Flower round and broadcasts only round metadata such as `server_round`, `split_layer`, and `max_seq_len`
2. each phone loads one local batch, runs the frozen prefix, and uploads:
   - `activation`
   - `target_embedding`
   - `attention_mask`
   - `target_token_ids`
   - `valid_lengths`
3. `server3` decodes the payload and feeds the uploaded activations into the Gemma suffix
4. `server3` computes the activation-space contrastive loss against the uploaded target embeddings
5. only the server-side LoRA weights are updated
6. the next client payload in the same round continues training the shared server suffix LoRA state

## What Makes This SplitLoRA

- the client/server cut is at `split_layer=0`
- the edge devices do not hold the suffix LoRA weights
- the trainable parameters are the server-side LoRA adapters on the split suffix
- the communication payload is split hidden-state data, not adapter tensors

So this is a split-learning LoRA baseline, not classic `FedAvg + LoRA`, `FedProx + LoRA`, or `FlexLoRA`.

## Loss

The local edge devices do not compute cross-entropy.

The server computes an activation-based contrastive objective:

- positive: the uploaded target embedding for the correct answer token
- negatives:
  - in-batch negatives
  - the server-side negative embedding queue
- metric:
  - multiple-choice accuracy from the suffix hidden state against `A/B/C/D` answer embeddings

## Checkpoints And Logs

- per-client rows are written to `server/metrics.csv`
- per-round summaries are written to `server/summary_rounds.csv`
- checkpoints are written under `server/checkpoints/`
- LoRA adapters are saved via `save_pretrained` alongside the main `.pt` checkpoint

## References

- SplitLoRA: [arXiv:2407.00952](https://arxiv.org/abs/2407.00952)
- SplitFed: [AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20819)
- LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
