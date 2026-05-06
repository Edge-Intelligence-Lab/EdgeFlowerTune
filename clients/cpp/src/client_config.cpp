#include "lshaped/client_config.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace lshaped {

namespace {

std::string RequireValue(const std::string& flag, int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
    }
    return argv[++index];
}

bool ParseBool(const std::string& value) {
    if (value == "1" || value == "true" || value == "True" || value == "TRUE") {
        return true;
    }
    if (value == "0" || value == "false" || value == "False" || value == "FALSE") {
        return false;
    }
    throw std::runtime_error("Invalid boolean value: " + value);
}

float ParseFloatValue(const std::string& flag, int argc, char** argv, int& index) {
    return std::stof(RequireValue(flag, argc, argv, index));
}

}  // namespace

ClientOptions ParseClientOptions(int argc, char** argv) {
    ClientOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--server_address") {
            options.server_address = RequireValue(arg, argc, argv, i);
        } else if (arg == "--client_id") {
            options.client_id = RequireValue(arg, argc, argv, i);
        } else if (arg == "--backend") {
            options.backend = RequireValue(arg, argc, argv, i);
        } else if (arg == "--model_dir") {
            options.model_dir = RequireValue(arg, argc, argv, i);
        } else if (arg == "--dataset_format") {
            options.dataset_format = RequireValue(arg, argc, argv, i);
        } else if (arg == "--dataset_csv") {
            options.dataset_csv = RequireValue(arg, argc, argv, i);
        } else if (arg == "--dataset_train_path") {
            options.dataset_train_path = RequireValue(arg, argc, argv, i);
        } else if (arg == "--dataset_valid_path") {
            options.dataset_valid_path = RequireValue(arg, argc, argv, i);
        } else if (arg == "--dataset_test_path") {
            options.dataset_test_path = RequireValue(arg, argc, argv, i);
        } else if (arg == "--run_mode") {
            options.run_mode = RequireValue(arg, argc, argv, i);
        } else if (arg == "--metrics_path") {
            options.metrics_path = RequireValue(arg, argc, argv, i);
        } else if (arg == "--client_index") {
            options.client_index = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--num_clients") {
            options.num_clients = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--batch_size") {
            options.batch_size = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--max_seq_len") {
            options.max_seq_len = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--max_rounds") {
            options.max_rounds = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--split_layer") {
            options.split_layer = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--local_steps") {
            options.local_steps = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--local_epochs") {
            options.local_epochs = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--grad_accum_steps") {
            options.grad_accum_steps = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--logging_steps") {
            options.logging_steps = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--lora_r") {
            options.lora_r = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--checkpoint_every") {
            options.checkpoint_every = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--checkpoint_mlp") {
            options.checkpoint_mlp = ParseBool(RequireValue(arg, argc, argv, i));
        } else if (arg == "--use_bf16_activations") {
            options.use_bf16_activations = ParseBool(RequireValue(arg, argc, argv, i));
        } else if (arg == "--mlp_chunk_size") {
            options.mlp_chunk_size = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--shard_max_resident_mb") {
            options.shard_max_resident_mb = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--shard_quantize_fp16_on_disk") {
            options.shard_quantize_fp16_on_disk = ParseBool(RequireValue(arg, argc, argv, i));
        } else if (arg == "--shard_quant_mode") {
            options.shard_quant_mode = RequireValue(arg, argc, argv, i);
        } else if (arg == "--shard_offload_dir") {
            options.shard_offload_dir = RequireValue(arg, argc, argv, i);
        } else if (arg == "--learning_rate") {
            options.learning_rate = ParseFloatValue(arg, argc, argv, i);
        } else if (arg == "--grad_clip_norm") {
            options.grad_clip_norm = ParseFloatValue(arg, argc, argv, i);
        } else if (arg == "--weight_decay") {
            options.weight_decay = ParseFloatValue(arg, argc, argv, i);
        } else if (arg == "--fedprox_mu") {
            options.fedprox_mu = ParseFloatValue(arg, argc, argv, i);
        } else if (arg == "--lora_alpha") {
            options.lora_alpha = ParseFloatValue(arg, argc, argv, i);
        } else if (arg == "--lora_dropout") {
            options.lora_dropout = ParseFloatValue(arg, argc, argv, i);
        } else if (arg == "--synthetic_samples") {
            options.synthetic_samples = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--grpc_max_message_mb") {
            options.grpc_max_message_mb = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--mock_hidden_size") {
            options.mock_hidden_size = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--connect_max_attempts") {
            options.connect_max_attempts = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--connect_ready_timeout_ms") {
            options.connect_ready_timeout_ms = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--connect_retry_delay_ms") {
            options.connect_retry_delay_ms = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--seed") {
            options.seed = std::stoi(RequireValue(arg, argc, argv, i));
        } else if (arg == "--verbose") {
            options.verbose = ParseBool(RequireValue(arg, argc, argv, i));
        } else if (arg == "--answer_prefix") {
            options.answer_prefix = RequireValue(arg, argc, argv, i);
        } else if (arg == "--target_mode") {
            options.target_mode = RequireValue(arg, argc, argv, i);
        } else if (arg == "--lora_targets") {
            options.lora_targets = RequireValue(arg, argc, argv, i);
        } else {
            throw std::runtime_error("Unknown flag: " + arg);
        }
    }

    if (options.client_id.empty()) {
        throw std::runtime_error("client_id must not be empty");
    }
    if (options.num_clients <= 0) {
        throw std::runtime_error("num_clients must be > 0");
    }
    if (options.client_index < 0 || options.client_index >= options.num_clients) {
        throw std::runtime_error("client_index must be in [0, num_clients)");
    }
    if (options.batch_size <= 0) {
        throw std::runtime_error("batch_size must be > 0");
    }
    if (options.connect_max_attempts <= 0) {
        throw std::runtime_error("connect_max_attempts must be > 0");
    }
    if (options.connect_ready_timeout_ms <= 0) {
        throw std::runtime_error("connect_ready_timeout_ms must be > 0");
    }
    if (options.connect_retry_delay_ms < 0) {
        throw std::runtime_error("connect_retry_delay_ms must be >= 0");
    }
    if (options.max_seq_len <= 0) {
        throw std::runtime_error("max_seq_len must be > 0");
    }
    if (options.split_layer < 0) {
        throw std::runtime_error("split_layer must be >= 0");
    }
    if (options.local_steps <= 0) {
        throw std::runtime_error("local_steps must be > 0");
    }
    if (options.local_epochs < 0) {
        throw std::runtime_error("local_epochs must be >= 0");
    }
    if (options.grad_accum_steps <= 0) {
        throw std::runtime_error("grad_accum_steps must be > 0");
    }
    if (options.logging_steps <= 0) {
        throw std::runtime_error("logging_steps must be > 0");
    }
    if (options.lora_r <= 0) {
        throw std::runtime_error("lora_r must be > 0");
    }
    if (options.checkpoint_every < 0) {
        throw std::runtime_error("checkpoint_every must be >= 0");
    }
    if (options.mlp_chunk_size < 0) {
        throw std::runtime_error("mlp_chunk_size must be >= 0");
    }
    if (options.shard_max_resident_mb < 0) {
        throw std::runtime_error("shard_max_resident_mb must be >= 0");
    }
    if (options.grad_clip_norm <= 0.0f) {
        throw std::runtime_error("grad_clip_norm must be > 0");
    }
    if (options.fedprox_mu < 0.0f) {
        throw std::runtime_error("fedprox_mu must be >= 0");
    }
    if (options.backend != "mock" && options.backend != "mft") {
        throw std::runtime_error("backend must be one of: mock, mft");
    }
    if (options.dataset_format != "mmlu_csv" && options.dataset_format != "wikitext_raw") {
        throw std::runtime_error("dataset_format must be one of: mmlu_csv, wikitext_raw");
    }
    return options;
}

void PrintClientOptions(const ClientOptions& options) {
    std::cout << "[client] server_address=" << options.server_address << "\n";
    std::cout << "[client] client_id=" << options.client_id << "\n";
    std::cout << "[client] backend=" << options.backend << "\n";
    std::cout << "[client] model_dir=" << options.model_dir << "\n";
    std::cout << "[client] dataset_format=" << options.dataset_format << "\n";
    std::cout << "[client] dataset_csv=" << options.dataset_csv << "\n";
    std::cout << "[client] dataset_train_path=" << options.dataset_train_path << "\n";
    std::cout << "[client] dataset_valid_path=" << options.dataset_valid_path << "\n";
    std::cout << "[client] dataset_test_path=" << options.dataset_test_path << "\n";
    std::cout << "[client] client_index=" << options.client_index << "\n";
    std::cout << "[client] num_clients=" << options.num_clients << "\n";
    std::cout << "[client] batch_size=" << options.batch_size << "\n";
    std::cout << "[client] max_seq_len=" << options.max_seq_len << "\n";
    std::cout << "[client] split_layer=" << options.split_layer << "\n";
    std::cout << "[client] local_steps=" << options.local_steps << "\n";
    std::cout << "[client] local_epochs=" << options.local_epochs << "\n";
    std::cout << "[client] grad_accum_steps=" << options.grad_accum_steps << "\n";
    std::cout << "[client] learning_rate=" << options.learning_rate << "\n";
    std::cout << "[client] grad_clip_norm=" << options.grad_clip_norm << "\n";
    std::cout << "[client] weight_decay=" << options.weight_decay << "\n";
    std::cout << "[client] fedprox_mu=" << options.fedprox_mu << "\n";
    std::cout << "[client] target_mode=" << options.target_mode << "\n";
    std::cout << "[client] lora_r=" << options.lora_r << "\n";
    std::cout << "[client] lora_alpha=" << options.lora_alpha << "\n";
    std::cout << "[client] lora_dropout=" << options.lora_dropout << "\n";
    std::cout << "[client] lora_targets=" << options.lora_targets << "\n";
    std::cout << "[client] checkpoint_every=" << options.checkpoint_every << "\n";
    std::cout << "[client] checkpoint_mlp=" << options.checkpoint_mlp << "\n";
    std::cout << "[client] use_bf16_activations=" << options.use_bf16_activations << "\n";
    std::cout << "[client] mlp_chunk_size=" << options.mlp_chunk_size << "\n";
    std::cout << "[client] shard_max_resident_mb=" << options.shard_max_resident_mb << "\n";
    std::cout << "[client] shard_quantize_fp16_on_disk=" << options.shard_quantize_fp16_on_disk << "\n";
    std::cout << "[client] shard_quant_mode=" << options.shard_quant_mode << "\n";
    std::cout << "[client] shard_offload_dir=" << options.shard_offload_dir << "\n";
    std::cout << "[client] synthetic_samples=" << options.synthetic_samples << "\n";
    std::cout << "[client] connect_max_attempts=" << options.connect_max_attempts << "\n";
    std::cout << "[client] connect_ready_timeout_ms=" << options.connect_ready_timeout_ms << "\n";
    std::cout << "[client] connect_retry_delay_ms=" << options.connect_retry_delay_ms << "\n";
}

}  // namespace lshaped
