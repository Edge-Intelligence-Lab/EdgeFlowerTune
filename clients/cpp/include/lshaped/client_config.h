#pragma once

#include <cstdint>
#include <string>

namespace lshaped {

struct ClientOptions {
    std::string server_address = "127.0.0.1:19080";
    std::string client_id = "client_0";
    std::string backend = "mock";  // "mock" or "mft"
    std::string model_dir;
    std::string dataset_format = "mmlu_csv";  // "mmlu_csv" or "wikitext_raw"
    std::string dataset_csv;
    std::string dataset_train_path;
    std::string dataset_valid_path;
    std::string dataset_test_path;
    std::string answer_prefix = " ";
    std::string run_mode = "train";
    std::string metrics_path;
    std::string target_mode = "attn";

    int client_index = 0;
    int num_clients = 1;
    int batch_size = 2;
    int max_seq_len = 128;
    int max_rounds = -1;
    int split_layer = 0;
    int local_steps = 1;
    int local_epochs = 0;
    int grad_accum_steps = 1;
    int logging_steps = 1;
    int lora_r = 8;
    int checkpoint_every = 0;
    int mlp_chunk_size = 0;
    int shard_max_resident_mb = 0;
    int synthetic_samples = 16;
    int grpc_max_message_mb = 128;
    int mock_hidden_size = 128;
    int connect_max_attempts = 8;
    int connect_ready_timeout_ms = 15000;
    int connect_retry_delay_ms = 5000;
    int seed = 7;
    float learning_rate = 2e-4f;
    float grad_clip_norm = 1.0f;
    float weight_decay = 0.0f;
    float fedprox_mu = 0.0f;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.0f;
    std::string lora_targets;
    std::string shard_quant_mode;
    std::string shard_offload_dir;
    bool use_bf16_activations = false;
    bool checkpoint_mlp = false;
    bool shard_quantize_fp16_on_disk = true;
    bool verbose = true;
};

ClientOptions ParseClientOptions(int argc, char** argv);
void PrintClientOptions(const ClientOptions& options);

}  // namespace lshaped
