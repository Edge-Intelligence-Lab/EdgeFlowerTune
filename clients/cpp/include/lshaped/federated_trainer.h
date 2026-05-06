#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "flwr/proto/transport.grpc.pb.h"

#include "lshaped/client_config.h"
#include "lshaped/mmlu_dataset.h"

namespace lshaped {

struct LocalFitSummary {
    std::int64_t num_examples = 0;
    std::int64_t steps_completed = 0;
    std::int64_t epochs_completed = 0;
    double mean_loss = 0.0;
    double mean_objective_loss = 0.0;
    double mean_prox_term = 0.0;
    double train_time_sec = 0.0;
    double avg_rss_mb = -1.0;
    double client_rss_mb = -1.0;
    double client_hwm_mb = -1.0;
    double client_swap_mb = -1.0;
    double mean_step_time_sec = 0.0;
    double max_step_time_sec = 0.0;
    std::uint64_t transmitted_bytes = 0;
    std::string step_times_sec_json;
};

class FederatedTrainer {
public:
    virtual ~FederatedTrainer() = default;

    virtual flwr::proto::Parameters GetParameters() const = 0;
    virtual const std::vector<std::string>& ParameterNames() const = 0;
    virtual std::string BackendName() const = 0;

    virtual LocalFitSummary Fit(
        const flwr::proto::Parameters& global_parameters,
        int batch_size,
        int max_seq_len,
        int local_steps,
        int local_epochs,
        float learning_rate,
        float weight_decay,
        int server_round) = 0;
};

std::unique_ptr<FederatedTrainer> CreateFederatedTrainer(
    const ClientOptions& options,
    ClientShardDataset dataset);

}  // namespace lshaped
