#include "lshaped/flower_legacy_client.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

namespace lshaped {

namespace {

constexpr std::int64_t kProtocolVersion = 3;

flwr::proto::Status MakeStatus(flwr::proto::Code code, const std::string& message) {
    flwr::proto::Status status;
    status.set_code(code);
    status.set_message(message);
    return status;
}

flwr::proto::Scalar MakeStringScalar(const std::string& value) {
    flwr::proto::Scalar scalar;
    scalar.set_string(value);
    return scalar;
}

flwr::proto::Scalar MakeIntScalar(std::int64_t value) {
    flwr::proto::Scalar scalar;
    scalar.set_sint64(value);
    return scalar;
}

flwr::proto::Scalar MakeUIntScalar(std::uint64_t value) {
    flwr::proto::Scalar scalar;
    scalar.set_uint64(value);
    return scalar;
}

flwr::proto::Scalar MakeDoubleScalar(double value) {
    flwr::proto::Scalar scalar;
    scalar.set_double_(value);
    return scalar;
}

bool HasSInt64(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kSint64;
}

bool HasUInt64(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kUint64;
}

bool HasDouble(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kDouble;
}

bool HasString(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kString;
}

std::int64_t ReadIntConfig(
    const google::protobuf::Map<std::string, flwr::proto::Scalar>& config,
    const std::string& key,
    std::int64_t fallback) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return fallback;
    }
    if (HasSInt64(it->second)) {
        return it->second.sint64();
    }
    if (HasUInt64(it->second)) {
        return static_cast<std::int64_t>(it->second.uint64());
    }
    throw std::runtime_error("Expected integer config for key: " + key);
}

double ReadDoubleConfig(
    const google::protobuf::Map<std::string, flwr::proto::Scalar>& config,
    const std::string& key,
    double fallback) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return fallback;
    }
    if (HasDouble(it->second)) {
        return it->second.double_();
    }
    if (HasSInt64(it->second)) {
        return static_cast<double>(it->second.sint64());
    }
    if (HasUInt64(it->second)) {
        return static_cast<double>(it->second.uint64());
    }
    throw std::runtime_error("Expected numeric config for key: " + key);
}

std::string JoinStrings(const std::vector<std::string>& items) {
    std::string joined;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i > 0) {
            joined += ",";
        }
        joined += items[i];
    }
    return joined;
}

bool IsRetryableStatus(const grpc::Status& status) {
    if (status.ok()) {
        return false;
    }
    switch (status.error_code()) {
        case grpc::StatusCode::UNAVAILABLE:
        case grpc::StatusCode::DEADLINE_EXCEEDED:
        case grpc::StatusCode::CANCELLED:
        case grpc::StatusCode::UNKNOWN:
            return true;
        default:
            return false;
    }
}

std::uint64_t ParameterBytes(const flwr::proto::Parameters& parameters) {
    std::uint64_t total = 0;
    for (const auto& tensor : parameters.tensors()) {
        total += static_cast<std::uint64_t>(tensor.size());
    }
    return total;
}

void AppendClientMetricLine(
    const ClientOptions& options,
    std::int64_t server_round,
    const LocalFitSummary& summary,
    double server_dispatch_ts,
    double fit_start_ts,
    double fit_end_ts,
    double response_ready_ts,
    double download_time_sec,
    double upload_time_sec,
    std::uint64_t download_bytes,
    std::uint64_t upload_bytes,
    std::size_t tensor_count) {
    if (options.metrics_path.empty()) {
        return;
    }
    const bool write_header = !std::ifstream(options.metrics_path).good();
    std::ofstream output(options.metrics_path, std::ios::app);
    if (!output) {
        throw std::runtime_error("Failed to open metrics_path for append: " + options.metrics_path);
    }
    if (write_header) {
        output << "client_id,server_round,num_examples,steps_completed,epochs_completed,mean_loss,"
               << "objective_loss,prox_term,train_time_sec,mean_step_time_sec,max_step_time_sec,"
               << "step_times_sec_json,avg_rss_mb,client_rss_mb,client_hwm_mb,client_swap_mb,"
               << "server_dispatch_ts,fit_start_ts,fit_end_ts,response_ready_ts,download_bytes,upload_bytes,"
               << "download_time_sec,upload_time_sec,client_download_read_time_sec,client_upload_write_time_sec,"
               << "transmitted_bytes,tensor_count\n";
    }
    output << options.client_id << ","
           << server_round << ","
           << summary.num_examples << ","
           << summary.steps_completed << ","
           << summary.epochs_completed << ","
           << summary.mean_loss << ","
           << summary.mean_objective_loss << ","
           << summary.mean_prox_term << ","
           << summary.train_time_sec << ","
           << summary.mean_step_time_sec << ","
           << summary.max_step_time_sec << ","
           << "\"" << summary.step_times_sec_json << "\"" << ","
           << summary.avg_rss_mb << ","
           << summary.client_rss_mb << ","
           << summary.client_hwm_mb << ","
           << summary.client_swap_mb << ","
           << server_dispatch_ts << ","
           << fit_start_ts << ","
           << fit_end_ts << ","
           << response_ready_ts << ","
           << download_bytes << ","
           << upload_bytes << ","
           << download_time_sec << ","
           << upload_time_sec << ","
           << download_time_sec << ","
           << upload_time_sec << ","
           << (download_bytes + upload_bytes) << ","
           << tensor_count << "\n";
}

}  // namespace

FlowerLegacyClient::FlowerLegacyClient(
    ClientOptions options,
    std::unique_ptr<FederatedTrainer> trainer)
    : options_(std::move(options)),
      trainer_(std::move(trainer)) {
    if (!trainer_) {
        throw std::runtime_error("trainer must not be null");
    }
}

int FlowerLegacyClient::Run() {
    grpc::ChannelArguments channel_arguments;
    const int max_message_bytes = options_.grpc_max_message_mb * 1024 * 1024;
    channel_arguments.SetMaxReceiveMessageSize(max_message_bytes);
    channel_arguments.SetMaxSendMessageSize(max_message_bytes);
    // Keep long-running Join streams alive with a low-frequency ping. Some
    // slower phones spend 30+ minutes inside local training with no
    // application-level traffic, which can trigger idle network middleboxes to
    // drop the stream before FitRes is uploaded. The Flower server used here
    // accepts pings at 10s+ intervals, so 60s is still conservative while
    // avoiding 20+ minute silent periods on slow phones.
    channel_arguments.SetInt("grpc.keepalive_time_ms", 60000);
    channel_arguments.SetInt("grpc.keepalive_timeout_ms", 30000);
    channel_arguments.SetInt("grpc.keepalive_permit_without_calls", 1);
    channel_arguments.SetInt("grpc.http2.max_pings_without_data", 0);
    channel_arguments.SetInt("grpc.http2.min_sent_ping_interval_without_data_ms", 60000);

    grpc::Status last_status;
    for (int attempt = 1; attempt <= options_.connect_max_attempts; ++attempt) {
        if (options_.verbose) {
            std::cout << "[client] connect_attempt=" << attempt
                      << "/" << options_.connect_max_attempts << "\n";
        }
        std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
            options_.server_address,
            grpc::InsecureChannelCredentials(),
            channel_arguments);
        const auto ready_deadline = std::chrono::system_clock::now() +
            std::chrono::milliseconds(options_.connect_ready_timeout_ms);
        if (!channel->WaitForConnected(ready_deadline)) {
            if (options_.verbose) {
                std::cerr << "[client] channel not ready before deadline on attempt "
                          << attempt << "\n";
            }
            if (attempt < options_.connect_max_attempts && options_.connect_retry_delay_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(options_.connect_retry_delay_ms));
            }
            continue;
        }

        auto stub = flwr::proto::FlowerService::NewStub(channel);
        grpc::ClientContext context;
        context.set_wait_for_ready(true);
        auto stream = stub->Join(&context);
        if (!stream) {
            last_status = grpc::Status(grpc::StatusCode::UNKNOWN, "Failed to create Flower Join stream");
        } else {
            bool saw_terminal_reconnect = false;
            flwr::proto::ServerMessage server_message;
            while (true) {
                const auto read_start = std::chrono::steady_clock::now();
                const bool read_ok = stream->Read(&server_message);
                last_message_read_time_sec_ = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - read_start).count();
                if (!read_ok) {
                    break;
                }
                flwr::proto::ClientMessage client_message = HandleServerMessage(server_message);
                const auto write_start = std::chrono::steady_clock::now();
                const bool write_ok = stream->Write(client_message);
                const double write_time_sec = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - write_start).count();
                if (!write_ok) {
                    std::cerr << "[client] stream write failed\n";
                    break;
                }
                if (server_message.msg_case() == flwr::proto::ServerMessage::kFitIns) {
                    FlushPendingFitMetrics(write_time_sec);
                }
                if (server_message.msg_case() == flwr::proto::ServerMessage::kReconnectIns) {
                    saw_terminal_reconnect = true;
                    break;
                }
            }

            stream->WritesDone();
            last_status = stream->Finish();
            if (last_status.ok()) {
                return 0;
            }
            if (saw_terminal_reconnect &&
                (last_status.error_code() == grpc::StatusCode::UNAVAILABLE ||
                 last_status.error_code() == grpc::StatusCode::CANCELLED)) {
                return 0;
            }
        }

        std::cerr << "[client] grpc status=" << last_status.error_code()
                  << " message=" << last_status.error_message()
                  << " attempt=" << attempt
                  << "/" << options_.connect_max_attempts << "\n";
        if (!IsRetryableStatus(last_status) || attempt == options_.connect_max_attempts) {
            return 1;
        }
        if (options_.connect_retry_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(options_.connect_retry_delay_ms));
        }
    }

    std::cerr << "[client] exhausted connect attempts without a successful Join stream\n";
    return 1;
}

void FlowerLegacyClient::FlushPendingFitMetrics(double upload_time_sec) {
    if (!pending_fit_metrics_.ready) {
        return;
    }
    AppendClientMetricLine(
        options_,
        pending_fit_metrics_.server_round,
        pending_fit_metrics_.summary,
        pending_fit_metrics_.server_dispatch_ts,
        pending_fit_metrics_.fit_start_ts,
        pending_fit_metrics_.fit_end_ts,
        pending_fit_metrics_.response_ready_ts,
        pending_fit_metrics_.download_time_sec,
        upload_time_sec,
        pending_fit_metrics_.download_bytes,
        pending_fit_metrics_.upload_bytes,
        pending_fit_metrics_.tensor_count);
    pending_fit_metrics_ = PendingFitMetrics{};
}

flwr::proto::ClientMessage FlowerLegacyClient::HandleServerMessage(
    const flwr::proto::ServerMessage& server_message) {
    switch (server_message.msg_case()) {
        case flwr::proto::ServerMessage::kGetPropertiesIns:
            return HandleGetProperties(server_message.get_properties_ins());
        case flwr::proto::ServerMessage::kGetParametersIns:
            return HandleGetParameters(server_message.get_parameters_ins());
        case flwr::proto::ServerMessage::kFitIns:
            return HandleFit(server_message.fit_ins());
        case flwr::proto::ServerMessage::kEvaluateIns: {
            flwr::proto::ClientMessage client_message;
            auto* evaluate_res = client_message.mutable_evaluate_res();
            *evaluate_res->mutable_status() = MakeStatus(
                flwr::proto::EVALUATE_NOT_IMPLEMENTED,
                "Classical FL client does not expose standalone evaluate");
            evaluate_res->set_loss(0.0f);
            evaluate_res->set_num_examples(0);
            return client_message;
        }
        case flwr::proto::ServerMessage::kReconnectIns:
            return HandleReconnect(server_message.reconnect_ins());
        case flwr::proto::ServerMessage::MSG_NOT_SET:
        default:
            throw std::runtime_error("Received unsupported Flower message");
    }
}

flwr::proto::ClientMessage FlowerLegacyClient::HandleGetProperties(
    const flwr::proto::ServerMessage::GetPropertiesIns& /*get_properties_ins*/) const {
    flwr::proto::ClientMessage client_message;
    auto* get_properties_res = client_message.mutable_get_properties_res();
    *get_properties_res->mutable_status() = MakeStatus(flwr::proto::OK, "OK");
    auto* properties = get_properties_res->mutable_properties();
    (*properties)["protocol_version"] = MakeIntScalar(kProtocolVersion);
    (*properties)["client_id"] = MakeStringScalar(options_.client_id);
    (*properties)["backend"] = MakeStringScalar(trainer_->BackendName());
    (*properties)["parameter_count"] = MakeUIntScalar(
        static_cast<std::uint64_t>(trainer_->ParameterNames().size()));
    (*properties)["parameter_names"] = MakeStringScalar(JoinStrings(trainer_->ParameterNames()));
    (*properties)["target_mode"] = MakeStringScalar(options_.target_mode);
    (*properties)["lora_r"] = MakeIntScalar(options_.lora_r);
    (*properties)["lora_alpha"] = MakeDoubleScalar(options_.lora_alpha);
    return client_message;
}

flwr::proto::ClientMessage FlowerLegacyClient::HandleGetParameters(
    const flwr::proto::ServerMessage::GetParametersIns& /*get_parameters_ins*/) const {
    flwr::proto::ClientMessage client_message;
    auto* get_parameters_res = client_message.mutable_get_parameters_res();
    *get_parameters_res->mutable_status() = MakeStatus(flwr::proto::OK, "OK");
    *get_parameters_res->mutable_parameters() = trainer_->GetParameters();
    return client_message;
}

flwr::proto::ClientMessage FlowerLegacyClient::HandleFit(
    const flwr::proto::ServerMessage::FitIns& fit_ins) {
    const double fit_start_ts = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::int64_t server_round = ReadIntConfig(fit_ins.config(), "server_round", -1);
    const double server_dispatch_ts = ReadDoubleConfig(fit_ins.config(), "server_dispatch_ts", 0.0);
    const int batch_size = static_cast<int>(ReadIntConfig(fit_ins.config(), "batch_size", options_.batch_size));
    const int max_seq_len = static_cast<int>(ReadIntConfig(fit_ins.config(), "max_seq_len", options_.max_seq_len));
    const int local_steps = static_cast<int>(ReadIntConfig(fit_ins.config(), "local_steps", options_.local_steps));
    const int local_epochs = static_cast<int>(ReadIntConfig(fit_ins.config(), "local_epochs", options_.local_epochs));
    const int grad_accum_steps = static_cast<int>(ReadIntConfig(
        fit_ins.config(), "grad_accum_steps", options_.grad_accum_steps));
    const float fedprox_mu = static_cast<float>(ReadDoubleConfig(
        fit_ins.config(), "prox_mu", static_cast<double>(options_.fedprox_mu)));
    const float learning_rate = static_cast<float>(ReadDoubleConfig(
        fit_ins.config(), "learning_rate", static_cast<double>(options_.learning_rate)));
    const float weight_decay = static_cast<float>(ReadDoubleConfig(
        fit_ins.config(), "weight_decay", static_cast<double>(options_.weight_decay)));

    options_.fedprox_mu = fedprox_mu;
    options_.grad_accum_steps = grad_accum_steps;
    const std::uint64_t download_bytes = ParameterBytes(fit_ins.parameters());

    const LocalFitSummary summary = trainer_->Fit(
        fit_ins.parameters(),
        batch_size,
        max_seq_len,
        local_steps,
        local_epochs,
        learning_rate,
        weight_decay,
        static_cast<int>(server_round));
    const double fit_end_ts = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    const flwr::proto::Parameters updated_parameters = trainer_->GetParameters();
    const double response_ready_ts = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::uint64_t upload_bytes = ParameterBytes(updated_parameters);
    const double download_time_sec = last_message_read_time_sec_;

    flwr::proto::ClientMessage client_message;
    auto* fit_res = client_message.mutable_fit_res();
    *fit_res->mutable_status() = MakeStatus(flwr::proto::OK, "OK");
    *fit_res->mutable_parameters() = updated_parameters;
    fit_res->set_num_examples(summary.num_examples);

    auto* metrics = fit_res->mutable_metrics();
    (*metrics)["protocol_version"] = MakeIntScalar(kProtocolVersion);
    (*metrics)["client_id"] = MakeStringScalar(options_.client_id);
    (*metrics)["backend"] = MakeStringScalar(trainer_->BackendName());
    (*metrics)["server_round"] = MakeIntScalar(server_round);
    (*metrics)["server_dispatch_ts"] = MakeDoubleScalar(server_dispatch_ts);
    (*metrics)["fit_start_ts"] = MakeDoubleScalar(fit_start_ts);
    (*metrics)["fit_end_ts"] = MakeDoubleScalar(fit_end_ts);
    (*metrics)["response_ready_ts"] = MakeDoubleScalar(response_ready_ts);
    (*metrics)["batch_size"] = MakeIntScalar(batch_size);
    (*metrics)["max_seq_len"] = MakeIntScalar(max_seq_len);
    (*metrics)["local_steps"] = MakeIntScalar(summary.steps_completed);
    (*metrics)["local_epochs"] = MakeIntScalar(summary.epochs_completed);
    (*metrics)["loss"] = MakeDoubleScalar(summary.mean_loss);
    (*metrics)["objective_loss"] = MakeDoubleScalar(summary.mean_objective_loss);
    (*metrics)["prox_term"] = MakeDoubleScalar(summary.mean_prox_term);
    (*metrics)["train_time_sec"] = MakeDoubleScalar(summary.train_time_sec);
    (*metrics)["mean_step_time_sec"] = MakeDoubleScalar(summary.mean_step_time_sec);
    (*metrics)["max_step_time_sec"] = MakeDoubleScalar(summary.max_step_time_sec);
    (*metrics)["step_times_sec_json"] = MakeStringScalar(summary.step_times_sec_json);
    (*metrics)["avg_rss_mb"] = MakeDoubleScalar(summary.avg_rss_mb);
    (*metrics)["client_rss_mb"] = MakeDoubleScalar(summary.client_rss_mb);
    (*metrics)["client_hwm_mb"] = MakeDoubleScalar(summary.client_hwm_mb);
    (*metrics)["client_swap_mb"] = MakeDoubleScalar(summary.client_swap_mb);
    (*metrics)["download_bytes"] = MakeUIntScalar(download_bytes);
    (*metrics)["upload_bytes"] = MakeUIntScalar(upload_bytes);
    (*metrics)["download_time_sec"] = MakeDoubleScalar(download_time_sec);
    (*metrics)["client_download_read_time_sec"] = MakeDoubleScalar(download_time_sec);
    (*metrics)["client_upload_write_time_sec"] = MakeDoubleScalar(0.0);
    (*metrics)["transmitted_bytes"] = MakeUIntScalar(download_bytes + upload_bytes);
    (*metrics)["parameter_count"] = MakeUIntScalar(
        static_cast<std::uint64_t>(trainer_->ParameterNames().size()));
    (*metrics)["parameter_names"] = MakeStringScalar(JoinStrings(trainer_->ParameterNames()));
    (*metrics)["target_mode"] = MakeStringScalar(options_.target_mode);
    (*metrics)["lora_r"] = MakeIntScalar(options_.lora_r);
    (*metrics)["lora_alpha"] = MakeDoubleScalar(options_.lora_alpha);
    (*metrics)["lora_dropout"] = MakeDoubleScalar(options_.lora_dropout);
    (*metrics)["prox_mu"] = MakeDoubleScalar(options_.fedprox_mu);

    pending_fit_metrics_.ready = true;
    pending_fit_metrics_.server_round = server_round;
    pending_fit_metrics_.summary = summary;
    pending_fit_metrics_.server_dispatch_ts = server_dispatch_ts;
    pending_fit_metrics_.fit_start_ts = fit_start_ts;
    pending_fit_metrics_.fit_end_ts = fit_end_ts;
    pending_fit_metrics_.response_ready_ts = response_ready_ts;
    pending_fit_metrics_.download_time_sec = download_time_sec;
    pending_fit_metrics_.download_bytes = download_bytes;
    pending_fit_metrics_.upload_bytes = upload_bytes;
    pending_fit_metrics_.tensor_count = trainer_->ParameterNames().size();

    if (options_.verbose) {
        std::cout << "[client] round=" << server_round
                  << " num_examples=" << summary.num_examples
                  << " loss=" << summary.mean_loss
                  << " train_s=" << summary.train_time_sec
                  << " bytes=" << (download_bytes + upload_bytes)
                  << " tensors=" << trainer_->ParameterNames().size()
                  << "\n";
    }
    ++local_round_id_;
    return client_message;
}

flwr::proto::ClientMessage FlowerLegacyClient::HandleReconnect(
    const flwr::proto::ServerMessage::ReconnectIns& reconnect_ins) {
    flwr::proto::ClientMessage client_message;
    auto* disconnect_res = client_message.mutable_disconnect_res();
    if (reconnect_ins.seconds() > 0) {
        disconnect_res->set_reason(flwr::proto::RECONNECT);
        if (options_.verbose) {
            std::cout << "[client] server requested reconnect after " << reconnect_ins.seconds() << "s\n";
        }
        std::this_thread::sleep_for(std::chrono::seconds(reconnect_ins.seconds()));
    } else {
        disconnect_res->set_reason(flwr::proto::ACK);
        if (options_.verbose) {
            std::cout << "[client] server closed stream with ACK\n";
        }
    }
    return client_message;
}

}  // namespace lshaped
