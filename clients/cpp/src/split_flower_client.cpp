#include "lshaped/split_flower_client.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

#include "lshaped/npy_serializer.h"

namespace lshaped {

namespace {

constexpr std::int64_t kProtocolVersion = 1;

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

bool HasString(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kString;
}

bool HasDouble(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kDouble;
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

std::string ReadStringConfig(
    const google::protobuf::Map<std::string, flwr::proto::Scalar>& config,
    const std::string& key,
    const std::string& fallback) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return fallback;
    }
    if (HasString(it->second)) {
        return it->second.string();
    }
    throw std::runtime_error("Expected string config for key: " + key);
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
    throw std::runtime_error("Expected floating-point config for key: " + key);
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

std::string JsonEncodeDoubles(const std::vector<double>& values) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

void AddNpyTensor(
    flwr::proto::Parameters* parameters,
    const void* data,
    std::size_t element_count,
    const std::vector<std::int64_t>& shape,
    NpyDType dtype) {
    std::vector<std::uint8_t> blob = SerializeNpy(data, element_count, shape, dtype);
    parameters->add_tensors(std::string(reinterpret_cast<const char*>(blob.data()), blob.size()));
}

std::uint64_t ParameterBytes(const flwr::proto::Parameters& parameters) {
    std::uint64_t total = 0;
    for (const auto& tensor : parameters.tensors()) {
        total += static_cast<std::uint64_t>(tensor.size());
    }
    return total;
}

double GetProcessRssMb() {
    std::ifstream input("/proc/self/status");
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("VmRSS:", 0) != 0) {
            continue;
        }
        std::istringstream iss(line.substr(6));
        long rss_kb = 0;
        iss >> rss_kb;
        if (rss_kb <= 0) {
            return -1.0;
        }
        return static_cast<double>(rss_kb) / 1024.0;
    }
    return -1.0;
}

void AppendBatch(EncodedBatch* dst, const EncodedBatch& src) {
    if (dst->batch_size == 0) {
        *dst = src;
        return;
    }
    if (dst->seq_len != src.seq_len || dst->hidden_size != src.hidden_size || dst->task_type != src.task_type) {
        throw std::runtime_error("Mismatched split batch shapes while appending local_steps payload");
    }
    dst->batch_size += src.batch_size;
    dst->activation.insert(dst->activation.end(), src.activation.begin(), src.activation.end());
    dst->target_embedding.insert(dst->target_embedding.end(), src.target_embedding.begin(), src.target_embedding.end());
    dst->attention_mask.insert(dst->attention_mask.end(), src.attention_mask.begin(), src.attention_mask.end());
    dst->target_token_ids.insert(dst->target_token_ids.end(), src.target_token_ids.begin(), src.target_token_ids.end());
    dst->valid_lengths.insert(dst->valid_lengths.end(), src.valid_lengths.begin(), src.valid_lengths.end());
    dst->answer_labels.insert(dst->answer_labels.end(), src.answer_labels.begin(), src.answer_labels.end());
}

void AppendSplitMetricLine(
    const ClientOptions& options,
    std::int64_t server_round,
    const EncodedBatch& batch,
    std::int64_t local_steps,
    const std::string& step_times_sec_json,
    double encode_time_sec,
    double serialize_time_sec,
    double round_time_sec,
    double avg_rss_mb,
    double peak_rss_mb,
    double server_dispatch_ts,
    double fit_start_ts,
    double fit_end_ts,
    double response_ready_ts,
    double download_time_sec,
    double upload_time_sec,
    std::uint64_t download_bytes,
    std::uint64_t upload_bytes) {
    if (options.metrics_path.empty()) {
        return;
    }
    const bool write_header = !std::ifstream(options.metrics_path).good();
    std::ofstream output(options.metrics_path, std::ios::app);
    if (!output) {
        throw std::runtime_error("Failed to open metrics_path for append: " + options.metrics_path);
    }
    if (write_header) {
        output << "client_id,server_round,batch_id,split_layer,batch_size,seq_len,hidden_size,"
               << "local_steps,step_times_sec_json,encode_time_sec,serialize_time_sec,round_time_sec,"
               << "avg_rss_mb,peak_rss_mb,server_dispatch_ts,fit_start_ts,fit_end_ts,response_ready_ts,"
               << "download_bytes,upload_bytes,download_time_sec,upload_time_sec,"
               << "client_download_read_time_sec,client_upload_write_time_sec,transmitted_bytes,client_rss_mb\n";
    }
    output << options.client_id << ","
           << server_round << ","
           << 0 << ","
           << options.split_layer << ","
           << batch.batch_size << ","
           << batch.seq_len << ","
           << batch.hidden_size << ","
           << local_steps << ","
           << "\"" << step_times_sec_json << "\"" << ","
           << encode_time_sec << ","
           << serialize_time_sec << ","
           << round_time_sec << ","
           << avg_rss_mb << ","
           << peak_rss_mb << ","
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
           << peak_rss_mb << "\n";
}

}  // namespace

SplitFlowerClient::SplitFlowerClient(
    ClientOptions options,
    ClientShardDataset dataset,
    std::unique_ptr<PrefixEncoder> encoder)
    : options_(std::move(options)),
      dataset_(std::move(dataset)),
      encoder_(std::move(encoder)) {
    if (!encoder_) {
        throw std::runtime_error("encoder must not be null");
    }
}

int SplitFlowerClient::Run() {
    grpc::ChannelArguments channel_arguments;
    const int max_message_bytes = options_.grpc_max_message_mb * 1024 * 1024;
    channel_arguments.SetMaxReceiveMessageSize(max_message_bytes);
    channel_arguments.SetMaxSendMessageSize(max_message_bytes);
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

void SplitFlowerClient::FlushPendingFitMetrics(double upload_time_sec) {
    if (!pending_fit_metrics_.ready) {
        return;
    }
    AppendSplitMetricLine(
        options_,
        pending_fit_metrics_.server_round,
        pending_fit_metrics_.batch,
        pending_fit_metrics_.local_steps,
        pending_fit_metrics_.step_times_sec_json,
        pending_fit_metrics_.encode_time_sec,
        pending_fit_metrics_.serialize_time_sec,
        pending_fit_metrics_.round_time_sec,
        pending_fit_metrics_.avg_rss_mb,
        pending_fit_metrics_.peak_rss_mb,
        pending_fit_metrics_.server_dispatch_ts,
        pending_fit_metrics_.fit_start_ts,
        pending_fit_metrics_.fit_end_ts,
        pending_fit_metrics_.response_ready_ts,
        pending_fit_metrics_.download_time_sec,
        upload_time_sec,
        pending_fit_metrics_.download_bytes,
        pending_fit_metrics_.upload_bytes);
    pending_fit_metrics_ = PendingFitMetrics{};
}

flwr::proto::ClientMessage SplitFlowerClient::HandleServerMessage(
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
                "Split client does not expose standalone evaluate");
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

flwr::proto::ClientMessage SplitFlowerClient::HandleGetProperties(
    const flwr::proto::ServerMessage::GetPropertiesIns& /*get_properties_ins*/) const {
    flwr::proto::ClientMessage client_message;
    auto* get_properties_res = client_message.mutable_get_properties_res();
    *get_properties_res->mutable_status() = MakeStatus(flwr::proto::OK, "OK");
    auto* properties = get_properties_res->mutable_properties();
    (*properties)["protocol_version"] = MakeIntScalar(kProtocolVersion);
    (*properties)["client_id"] = MakeStringScalar(options_.client_id);
    (*properties)["backend"] = MakeStringScalar(options_.backend);
    (*properties)["run_mode"] = MakeStringScalar(options_.run_mode);
    (*properties)["split_layer"] = MakeIntScalar(options_.split_layer);
    return client_message;
}

flwr::proto::ClientMessage SplitFlowerClient::HandleGetParameters(
    const flwr::proto::ServerMessage::GetParametersIns& /*get_parameters_ins*/) const {
    flwr::proto::ClientMessage client_message;
    auto* get_parameters_res = client_message.mutable_get_parameters_res();
    *get_parameters_res->mutable_status() = MakeStatus(flwr::proto::OK, "OK");
    auto* parameters = get_parameters_res->mutable_parameters();
    parameters->set_tensor_type("numpy.ndarray");
    AddNpyTensor(parameters, &kProtocolVersion, 1, {1}, NpyDType::kInt32);
    return client_message;
}

flwr::proto::ClientMessage SplitFlowerClient::HandleFit(
    const flwr::proto::ServerMessage::FitIns& fit_ins) {
    const double fit_start_ts = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::int64_t server_round = ReadIntConfig(fit_ins.config(), "server_round", -1);
    const double server_dispatch_ts = ReadDoubleConfig(fit_ins.config(), "server_dispatch_ts", 0.0);
    const std::string mode = ReadStringConfig(fit_ins.config(), "mode", "train");
    const int requested_local_steps = static_cast<int>(ReadIntConfig(fit_ins.config(), "local_steps", options_.local_steps));
    options_.max_seq_len = static_cast<int>(ReadIntConfig(fit_ins.config(), "max_seq_len", options_.max_seq_len));
    options_.batch_size = static_cast<int>(ReadIntConfig(fit_ins.config(), "batch_size", options_.batch_size));
    options_.split_layer = static_cast<int>(ReadIntConfig(fit_ins.config(), "split_layer", options_.split_layer));
    const std::uint64_t download_bytes = ParameterBytes(fit_ins.parameters());

    const auto round_start = std::chrono::steady_clock::now();
    EncodedBatch batch{};
    std::vector<double> step_times_sec;
    std::vector<double> rss_samples_mb;
    int effective_local_steps = std::max(1, requested_local_steps);
    for (int step = 0; step < effective_local_steps; ++step) {
        const auto step_start = std::chrono::steady_clock::now();
        EncodedBatch step_batch{};
        if (options_.dataset_format == "wikitext_raw") {
            step_batch = encoder_->EncodeNextBatch(options_);
        } else {
            std::vector<MMLUSample> batch_samples = dataset_.NextBatch(options_.batch_size);
            step_batch = encoder_->Encode(batch_samples, options_);
        }
        AppendBatch(&batch, step_batch);
        const auto step_end = std::chrono::steady_clock::now();
        step_times_sec.push_back(std::chrono::duration<double>(step_end - step_start).count());
        const double rss_mb = GetProcessRssMb();
        if (rss_mb > 0.0) {
            rss_samples_mb.push_back(rss_mb);
        }
    }
    const auto encode_end = std::chrono::steady_clock::now();
    const double fit_end_ts = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    flwr::proto::ClientMessage client_message;
    auto* fit_res = client_message.mutable_fit_res();
    *fit_res->mutable_status() = MakeStatus(flwr::proto::OK, "OK");

    const auto serialize_start = std::chrono::steady_clock::now();
    auto* parameters = fit_res->mutable_parameters();
    parameters->set_tensor_type("numpy.ndarray");
    AddNpyTensor(
        parameters,
        batch.activation.data(),
        batch.activation.size(),
        {batch.batch_size, batch.seq_len, batch.hidden_size},
        NpyDType::kFloat32);
    AddNpyTensor(
        parameters,
        batch.target_embedding.data(),
        batch.target_embedding.size(),
        {batch.batch_size, batch.hidden_size},
        NpyDType::kFloat32);
    AddNpyTensor(
        parameters,
        batch.attention_mask.data(),
        batch.attention_mask.size(),
        {batch.batch_size, batch.seq_len},
        NpyDType::kInt32);
    AddNpyTensor(
        parameters,
        batch.target_token_ids.data(),
        batch.target_token_ids.size(),
        batch.task_type == "next_token_lm"
            ? std::vector<std::int64_t>{batch.batch_size, batch.seq_len}
            : std::vector<std::int64_t>{batch.batch_size},
        NpyDType::kInt32);
    AddNpyTensor(
        parameters,
        batch.valid_lengths.data(),
        batch.valid_lengths.size(),
        {batch.batch_size},
        NpyDType::kInt32);
    const auto serialize_end = std::chrono::steady_clock::now();
    const double response_ready_ts = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::uint64_t upload_bytes = batch.transmitted_bytes();

    fit_res->set_num_examples(batch.batch_size);
    auto* metrics = fit_res->mutable_metrics();
    const double encode_time_sec = std::chrono::duration<double>(encode_end - round_start).count();
    const double serialize_time_sec = std::chrono::duration<double>(serialize_end - serialize_start).count();
    const double round_time_sec = std::chrono::duration<double>(serialize_end - round_start).count();
    const double avg_rss_mb = rss_samples_mb.empty()
        ? -1.0
        : std::accumulate(rss_samples_mb.begin(), rss_samples_mb.end(), 0.0) /
            static_cast<double>(rss_samples_mb.size());
    double peak_rss_mb = -1.0;
    for (double value : rss_samples_mb) {
        peak_rss_mb = std::max(peak_rss_mb, value);
    }
    const std::string step_times_json = JsonEncodeDoubles(step_times_sec);
    const double download_time_sec = last_message_read_time_sec_;
    (*metrics)["protocol_version"] = MakeIntScalar(kProtocolVersion);
    (*metrics)["client_id"] = MakeStringScalar(options_.client_id);
    (*metrics)["batch_id"] = MakeIntScalar(batch_id_);
    (*metrics)["mode"] = MakeStringScalar(mode);
    (*metrics)["task_type"] = MakeStringScalar(batch.task_type);
    (*metrics)["split_layer"] = MakeIntScalar(options_.split_layer);
    (*metrics)["activation_shape"] = MakeStringScalar(
        std::to_string(batch.batch_size) + "x" + std::to_string(batch.seq_len) + "x" + std::to_string(batch.hidden_size));
    (*metrics)["activation_dtype"] = MakeStringScalar("float32");
    (*metrics)["target_shape"] = MakeStringScalar(
        std::to_string(batch.batch_size) + "x" + std::to_string(batch.hidden_size));
    (*metrics)["target_dtype"] = MakeStringScalar("float32");
    (*metrics)["attention_shape"] = MakeStringScalar(
        std::to_string(batch.batch_size) + "x" + std::to_string(batch.seq_len));
    (*metrics)["attention_dtype"] = MakeStringScalar("int32");
    (*metrics)["target_token_ids_shape"] = MakeStringScalar(
        batch.task_type == "next_token_lm"
            ? std::to_string(batch.batch_size) + "x" + std::to_string(batch.seq_len)
            : std::to_string(batch.batch_size));
    (*metrics)["answer_labels"] = MakeStringScalar(JoinStrings(batch.answer_labels));
    (*metrics)["transmitted_bytes"] = MakeUIntScalar(batch.transmitted_bytes());
    (*metrics)["local_steps"] = MakeIntScalar(effective_local_steps);
    (*metrics)["step_batch_size"] = MakeIntScalar(options_.batch_size);
    (*metrics)["retry_count"] = MakeIntScalar(0);
    (*metrics)["server_round"] = MakeIntScalar(server_round);
    (*metrics)["client_backend"] = MakeStringScalar(options_.backend);
    (*metrics)["server_dispatch_ts"] = MakeDoubleScalar(server_dispatch_ts);
    (*metrics)["fit_start_ts"] = MakeDoubleScalar(fit_start_ts);
    (*metrics)["fit_end_ts"] = MakeDoubleScalar(fit_end_ts);
    (*metrics)["response_ready_ts"] = MakeDoubleScalar(response_ready_ts);
    (*metrics)["client_encode_time_sec"] = MakeDoubleScalar(encode_time_sec);
    (*metrics)["client_serialize_time_sec"] = MakeDoubleScalar(serialize_time_sec);
    (*metrics)["client_round_time_sec"] = MakeDoubleScalar(round_time_sec);
    (*metrics)["step_times_sec_json"] = MakeStringScalar(step_times_json);
    (*metrics)["avg_rss_mb"] = MakeDoubleScalar(avg_rss_mb);
    (*metrics)["peak_rss_mb"] = MakeDoubleScalar(peak_rss_mb);
    (*metrics)["client_rss_mb"] = MakeDoubleScalar(peak_rss_mb);
    (*metrics)["download_bytes"] = MakeUIntScalar(download_bytes);
    (*metrics)["upload_bytes"] = MakeUIntScalar(upload_bytes);
    (*metrics)["download_time_sec"] = MakeDoubleScalar(download_time_sec);
    (*metrics)["client_download_read_time_sec"] = MakeDoubleScalar(download_time_sec);
    (*metrics)["client_upload_write_time_sec"] = MakeDoubleScalar(0.0);
    (*metrics)["transmitted_bytes"] = MakeUIntScalar(download_bytes + upload_bytes);
    (*metrics)["client_power_w"] = MakeDoubleScalar(-1.0);

    pending_fit_metrics_.ready = true;
    pending_fit_metrics_.server_round = server_round;
    pending_fit_metrics_.batch = batch;
    pending_fit_metrics_.local_steps = effective_local_steps;
    pending_fit_metrics_.step_times_sec_json = step_times_json;
    pending_fit_metrics_.encode_time_sec = encode_time_sec;
    pending_fit_metrics_.serialize_time_sec = serialize_time_sec;
    pending_fit_metrics_.round_time_sec = round_time_sec;
    pending_fit_metrics_.avg_rss_mb = avg_rss_mb;
    pending_fit_metrics_.peak_rss_mb = peak_rss_mb;
    pending_fit_metrics_.server_dispatch_ts = server_dispatch_ts;
    pending_fit_metrics_.fit_start_ts = fit_start_ts;
    pending_fit_metrics_.fit_end_ts = fit_end_ts;
    pending_fit_metrics_.response_ready_ts = response_ready_ts;
    pending_fit_metrics_.download_time_sec = download_time_sec;
    pending_fit_metrics_.download_bytes = download_bytes;
    pending_fit_metrics_.upload_bytes = upload_bytes;
    ++batch_id_;
    return client_message;
}

flwr::proto::ClientMessage SplitFlowerClient::HandleReconnect(
    const flwr::proto::ServerMessage::ReconnectIns& reconnect_ins) {
    flwr::proto::ClientMessage client_message;
    auto* disconnect_res = client_message.mutable_disconnect_res();
    if (reconnect_ins.seconds() > 0) {
        if (options_.verbose) {
            std::cout << "[client] sleeping " << reconnect_ins.seconds()
                      << " second(s) before reconnect\n";
        }
        std::this_thread::sleep_for(std::chrono::seconds(reconnect_ins.seconds()));
    } else if (options_.verbose) {
        std::cout << "[client] server closed stream with ACK\n";
    }
    disconnect_res->set_reason(flwr::proto::ACK);
    return client_message;
}

}  // namespace lshaped
