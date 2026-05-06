#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "flwr/proto/transport.grpc.pb.h"

#include "lshaped/client_config.h"
#include "lshaped/federated_trainer.h"

namespace lshaped {

class FlowerLegacyClient {
public:
    FlowerLegacyClient(ClientOptions options, std::unique_ptr<FederatedTrainer> trainer);

    int Run();

private:
    struct PendingFitMetrics {
        bool ready = false;
        std::int64_t server_round = 0;
        LocalFitSummary summary;
        double server_dispatch_ts = 0.0;
        double fit_start_ts = 0.0;
        double fit_end_ts = 0.0;
        double response_ready_ts = 0.0;
        double download_time_sec = 0.0;
        std::uint64_t download_bytes = 0;
        std::uint64_t upload_bytes = 0;
        std::size_t tensor_count = 0;
    };

    ClientOptions options_;
    std::unique_ptr<FederatedTrainer> trainer_;
    std::int64_t local_round_id_ = 0;
    double last_message_read_time_sec_ = 0.0;
    PendingFitMetrics pending_fit_metrics_;

    flwr::proto::ClientMessage HandleServerMessage(const flwr::proto::ServerMessage& server_message);
    flwr::proto::ClientMessage HandleGetProperties(
        const flwr::proto::ServerMessage::GetPropertiesIns& get_properties_ins) const;
    flwr::proto::ClientMessage HandleGetParameters(
        const flwr::proto::ServerMessage::GetParametersIns& get_parameters_ins) const;
    flwr::proto::ClientMessage HandleFit(const flwr::proto::ServerMessage::FitIns& fit_ins);
    flwr::proto::ClientMessage HandleReconnect(
        const flwr::proto::ServerMessage::ReconnectIns& reconnect_ins);
    void FlushPendingFitMetrics(double upload_time_sec);
};

}  // namespace lshaped
