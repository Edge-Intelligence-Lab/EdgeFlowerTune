# Legacy Split Archive

This directory stores the previous split-FL experiment path as a backup.

Archived here:

- split/server-side LoRA YAML configs
- mock/split communication smoke configs
- split-only orchestration scripts
- split-only documentation
- split Python client/server source modules

Active development should not import defaults from this directory.

The active classic path stays in:

- [configs](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs)
- [scripts](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/scripts)
- [docs](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/docs)

Current baseline:

- each edge device trains Gemma-270M LoRA locally
- each edge device uploads adapter tensors to Flower
- `server3` performs weighted `FedAvg`
