#include "gemma_trainer.h"
#include "../data/wikitext2_dataset.h"
#include "../core/lm_loss.h"
#include "../core/ops.h"
#include "../core/logger.h"
#include "../core/memory_manager.h"
#include "../core/performance_monitor.h"
#include "../../opt_ops/energy/power_monitor.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

namespace ops {

namespace {

std::string make_checkpoint_path(const std::string& output_dir, int step) {
    std::ostringstream ss;
    ss << output_dir;
    if (!output_dir.empty() && output_dir.back() != '/') {
        ss << "/";
    }
    ss << "gemma_lora_step" << step << ".safetensors";
    return ss.str();
}

void check_tensor_finite_or_throw(const TensorPtr& tensor,
                                  const std::string& stage) {
    const char* enabled = std::getenv("OPS_GEMMA_FINITE_CHECK");
    if (!enabled || std::string(enabled) != "1") {
        return;
    }
    if (!tensor) {
        throw std::runtime_error("Finite check failed: null tensor at " + stage);
    }
    if (tensor->dtype() != kFloat32 && tensor->dtype() != kFloat16) {
        return;
    }
    TensorPtr fp32 = tensor;
    if (tensor->dtype() != kFloat32) {
        fp32 = cast(tensor, kFloat32);
    }
    const float* data = fp32->data<float>();
    const int64_t n = fp32->numel();
    for (int64_t i = 0; i < n; ++i) {
        const float v = data[i];
        if (!std::isfinite(v)) {
            std::ostringstream oss;
            oss << "[GemmaFiniteCheck] non-finite tensor at " << stage
                << " numel=" << n
                << " bad_index=" << i
                << " value=" << v;
            std::cerr << oss.str() << std::endl;
            throw std::runtime_error(oss.str());
        }
    }
}

bool gemma_trace_enabled() {
    const char* enabled = std::getenv("OPS_GEMMA_TRACE_PROFILE");
    return enabled && std::string(enabled) == "1";
}

int gemma_trace_limit() {
    const char* raw = std::getenv("OPS_GEMMA_TRACE_LIMIT");
    if (!raw) return 16;
    try {
        return std::max(1, std::stoi(raw));
    } catch (...) {
        return 16;
    }
}

}  // namespace

GemmaLoRATrainer::GemmaLoRATrainer(GemmaModel& model,
                                   GemmaLoraInjector& injector,
                                   WikiText2Dataset& train_data,
                                   WikiText2Dataset& eval_data,
                                   const GemmaTrainerConfig& config)
    : model_(model),
      injector_(injector),
      train_data_(train_data),
      eval_data_(eval_data),
      config_(config),
      power_monitor_(),
      global_step_(0) {

    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.beta1 = config_.adam_beta1;
    adam_cfg.beta2 = config_.adam_beta2;
    adam_cfg.epsilon = config_.adam_eps;
    adam_cfg.weight_decay = config_.weight_decay;
    adam_cfg.clip_grad_norm = config_.max_grad_norm;

    optimizer_ = std::make_unique<Adam>(adam_cfg);

    // Init PowerMonitor
    energy::PowerConfig pm_cfg;
    pm_cfg.check_interval_steps = config_.pm_interval;
    pm_cfg.battery_threshold = config_.pm_batt_thresh;
    pm_cfg.temp_threshold = config_.pm_temp_thresh;
    pm_cfg.freq_b_high = config_.pm_fb_high;
    pm_cfg.freq_b_low = config_.pm_fb_low;
    pm_cfg.freq_t_high = config_.pm_ft_high;
    pm_cfg.freq_t_low = config_.pm_ft_low;
    pm_cfg.enable_battery = config_.pm_enable_batt;
    pm_cfg.enable_temp = config_.pm_enable_temp;
    power_monitor_ = energy::PowerMonitor(pm_cfg);
    power_monitor_.set_manual_readings(config_.pm_manual_batt, config_.pm_manual_temp);
    if (!config_.pm_schedule.empty()) {
        power_monitor_.set_step_schedule(energy::PowerMonitor::parse_schedule(config_.pm_schedule));
    }

    if (config_.fedprox_mu > 0.0f) {
        const auto trainable = injector_.get_trainable_params();
        proximal_reference_.reserve(trainable.size());
        for (const auto& tensor : trainable) {
            if (!tensor) {
                proximal_reference_.push_back({});
                continue;
            }
            const float* data = tensor->data<float>();
            proximal_reference_.emplace_back(data, data + tensor->numel());
        }
    }

    std::cout << "[GemmaTrainer] LR=" << config_.learning_rate
              << ", epochs=" << config_.num_epochs
              << ", grad_accum=" << config_.grad_accum_steps
              << ", fedprox_mu=" << config_.fedprox_mu << std::endl;
}

float GemmaLoRATrainer::get_lr(int step) {
    int64_t micro_per_epoch = (train_data_.num_sequences() + config_.micro_batch_size - 1) / config_.micro_batch_size;
    int64_t total_updates = (micro_per_epoch * config_.num_epochs + config_.grad_accum_steps - 1) / config_.grad_accum_steps;
    total_updates = std::max<int64_t>(1, total_updates);

    int warmup_steps = static_cast<int>(total_updates * config_.warmup_ratio);

    if (warmup_steps > 0 && step <= warmup_steps) {
        return config_.learning_rate * (static_cast<float>(step) / warmup_steps);
    }

    float progress = static_cast<float>(step - warmup_steps) / std::max<int64_t>(1, total_updates - warmup_steps);
    progress = std::clamp(progress, 0.0f, 1.0f);

    if (config_.lr_scheduler == "cosine") {
        return config_.learning_rate * 0.5f * (1.0f + std::cos(3.14159265f * progress));
    }

    return config_.learning_rate * (1.0f - progress);
}

void GemmaLoRATrainer::clip_gradients() {
    auto params = injector_.get_trainable_params();
    float total_norm = 0.0f;
    for (const auto& param : params) {
        if (!param->grad()) continue;
        const float* g = param->grad()->data<float>();
        for (int64_t i = 0; i < param->grad()->numel(); ++i) {
            total_norm += g[i] * g[i];
        }
    }
    total_norm = std::sqrt(total_norm);
    if (total_norm <= config_.max_grad_norm) return;
    float scale = config_.max_grad_norm / (total_norm + 1e-6f);
    for (const auto& param : params) {
        if (!param->grad()) continue;
        float* g = param->grad()->data<float>();
        for (int64_t i = 0; i < param->grad()->numel(); ++i) g[i] *= scale;
    }
}

float GemmaLoRATrainer::apply_fedprox_gradients() {
    last_prox_term_ = 0.0f;
    if (config_.fedprox_mu <= 0.0f || proximal_reference_.empty()) {
        return 0.0f;
    }

    const auto params = injector_.get_trainable_params();
    if (params.size() != proximal_reference_.size()) {
        throw std::runtime_error("FedProx reference count mismatch");
    }

    for (std::size_t index = 0; index < params.size(); ++index) {
        const auto& param = params[index];
        if (!param) {
            continue;
        }
        if (!param->grad()) {
            param->set_grad(zeros(param->shape(), param->dtype(), param->device()));
        }
        float* grad = param->grad()->data<float>();
        const float* data = param->data<float>();
        const auto& ref = proximal_reference_[index];
        if (ref.size() != static_cast<std::size_t>(param->numel())) {
            throw std::runtime_error("FedProx reference shape mismatch");
        }
        for (int64_t i = 0; i < param->numel(); ++i) {
            const float diff = data[i] - ref[static_cast<std::size_t>(i)];
            grad[i] += config_.fedprox_mu * diff;
            last_prox_term_ += 0.5f * config_.fedprox_mu * diff * diff;
        }
    }
    return last_prox_term_;
}

float GemmaLoRATrainer::train_step(const Batch& batch) {
    auto rss_mb = []() -> double {
        return static_cast<double>(MemoryMonitor::get_system_memory_usage()) / (1024.0 * 1024.0);
    };

    double rss_pre_mb = rss_mb();
    micro_step_counter_++;
    if (config_.dump_embedding && !embedding_dump_scheduled_ &&
        micro_step_counter_ == config_.dump_embedding_step) {
        model_.request_embedding_dump(config_.dump_embedding_step, config_.dump_embedding_dir);
        embedding_dump_scheduled_ = true;
    }

    auto step_t0 = std::chrono::steady_clock::now();
    auto logits = model_.forward(batch.input_ids, batch.attention_mask);
    auto step_t1 = std::chrono::steady_clock::now();
    check_tensor_finite_or_throw(logits, "trainer/logits");
    auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
    auto step_t2 = std::chrono::steady_clock::now();
    check_tensor_finite_or_throw(loss, "trainer/loss");
    float loss_val = loss->data<float>()[0];
    double rss_fwd_mb = rss_mb();

    float scale = 1.0f / static_cast<float>(config_.grad_accum_steps);
    float total_scale = scale * config_.loss_scale;
    auto scaled_loss = mul(loss, total_scale);
    scaled_loss->backward();
    auto step_t3 = std::chrono::steady_clock::now();
    double rss_bwd_mb = rss_mb();

    accum_counter_++;
    accum_loss_ += loss_val;

    if (accum_counter_ < config_.grad_accum_steps) {
        double step_max_mb = std::max({rss_pre_mb, rss_fwd_mb, rss_bwd_mb});
        peak_rss_mb_ = std::max(peak_rss_mb_, step_max_mb);
        last_rss_pre_mb_ = rss_pre_mb;
        last_rss_fwd_mb_ = rss_fwd_mb;
        last_rss_bwd_mb_ = rss_bwd_mb;
        last_rss_opt_mb_ = rss_bwd_mb;
        last_rss_post_mb_ = rss_bwd_mb;
        last_step_max_rss_mb_ = step_max_mb;
        if (gemma_trace_enabled()) {
            static int trace_count = 0;
            if (trace_count < gemma_trace_limit()) {
                ++trace_count;
                const auto forward_ms = std::chrono::duration_cast<std::chrono::milliseconds>(step_t1 - step_t0).count();
                const auto loss_ms = std::chrono::duration_cast<std::chrono::milliseconds>(step_t2 - step_t1).count();
                const auto backward_ms = std::chrono::duration_cast<std::chrono::milliseconds>(step_t3 - step_t2).count();
                std::cout << "[GemmaTrace] micro_step"
                          << " step=" << micro_step_counter_
                          << " accum=" << accum_counter_ << "/" << config_.grad_accum_steps
                          << " batch=" << batch.input_ids->shape()[0]
                          << " seq=" << batch.input_ids->shape()[1]
                          << " loss=" << loss_val
                          << " forward_ms=" << forward_ms
                          << " loss_ms=" << loss_ms
                          << " backward_ms=" << backward_ms
                          << " rss_pre_mb=" << rss_pre_mb
                          << " rss_fwd_mb=" << rss_fwd_mb
                          << " rss_bwd_mb=" << rss_bwd_mb
                          << std::endl;
            }
        }
        return -1.0f;
    }

    if (config_.loss_scale != 1.0f) {
        float inv_scale = 1.0f / config_.loss_scale;
        auto params = injector_.get_trainable_params();
        for (const auto& param : params) {
            if (!param->grad()) continue;
            float* g = param->grad()->data<float>();
            for (int64_t i = 0; i < param->grad()->numel(); ++i) {
                g[i] *= inv_scale;
            }
        }
    }

    const float prox_term = apply_fedprox_gradients();
    clip_gradients();

    auto params = injector_.get_trainable_params();
    std::vector<TensorPtr> grads;
    grads.reserve(params.size());
    for (const auto& param : params) {
        grads.push_back(param->grad());
    }

    global_step_++;
    float current_lr = get_lr(global_step_);
    optimizer_->set_learning_rate(current_lr);
    optimizer_->step(params, grads);
    auto step_t4 = std::chrono::steady_clock::now();
    double rss_opt_mb = rss_mb();
    for (const auto& param : params) {
        if (param->grad()) param->zero_grad();
    }
    MemoryManager::instance().force_cleanup();
    auto step_t5 = std::chrono::steady_clock::now();
    double rss_post_mb = rss_mb();

    double step_max_mb = std::max({rss_pre_mb, rss_fwd_mb, rss_bwd_mb, rss_opt_mb, rss_post_mb});
    peak_rss_mb_ = std::max(peak_rss_mb_, step_max_mb);
    last_rss_pre_mb_ = rss_pre_mb;
    last_rss_fwd_mb_ = rss_fwd_mb;
    last_rss_bwd_mb_ = rss_bwd_mb;
    last_rss_opt_mb_ = rss_opt_mb;
    last_rss_post_mb_ = rss_post_mb;
    last_step_max_rss_mb_ = step_max_mb;

    accum_counter_ = 0;
    float avg_loss = accum_loss_ / static_cast<float>(config_.grad_accum_steps);
    accum_loss_ = 0.0f;
    if (gemma_trace_enabled()) {
        static int trace_count = 0;
        if (trace_count < gemma_trace_limit()) {
            ++trace_count;
            const auto forward_ms = std::chrono::duration_cast<std::chrono::milliseconds>(step_t1 - step_t0).count();
            const auto loss_ms = std::chrono::duration_cast<std::chrono::milliseconds>(step_t2 - step_t1).count();
            const auto backward_ms = std::chrono::duration_cast<std::chrono::milliseconds>(step_t3 - step_t2).count();
            const auto optimizer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(step_t4 - step_t3).count();
            const auto cleanup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(step_t5 - step_t4).count();
            std::cout << "[GemmaTrace] optimizer_step"
                      << " global_step=" << global_step_
                      << " micro_step=" << micro_step_counter_
                      << " batch=" << batch.input_ids->shape()[0]
                      << " seq=" << batch.input_ids->shape()[1]
                      << " avg_loss=" << avg_loss
                      << " forward_ms=" << forward_ms
                      << " loss_ms=" << loss_ms
                      << " backward_ms=" << backward_ms
                      << " optimizer_ms=" << optimizer_ms
                      << " cleanup_ms=" << cleanup_ms
                      << " rss_pre_mb=" << rss_pre_mb
                      << " rss_fwd_mb=" << rss_fwd_mb
                      << " rss_bwd_mb=" << rss_bwd_mb
                      << " rss_opt_mb=" << rss_opt_mb
                      << " rss_post_mb=" << rss_post_mb
                      << std::endl;
        }
    }
    return avg_loss + prox_term;
}

float GemmaLoRATrainer::evaluate() {
    std::cout << "[GemmaTrainer] Eval started..." << std::endl;
    eval_data_.reset_cursor();
    float total_loss = 0.0f;
    int batches = 0;

    while (true) {
        auto batch = eval_data_.next_batch(config_.micro_batch_size, false);
        if (!batch.input_ids) break;
        auto logits = model_.forward(batch.input_ids, batch.attention_mask);
        auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
        total_loss += loss->data<float>()[0];
        batches++;
    }

    float mean_loss = (batches > 0) ? total_loss / batches : 0.0f;
    float ppl = perplexity_from_loss(mean_loss);
    std::cout << "  Eval Loss: " << mean_loss << " PPL: " << ppl << std::endl;
    MemoryManager::instance().force_cleanup();
    return mean_loss;
}

void GemmaLoRATrainer::train() {
    for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
        std::cout << "\n=== Gemma Epoch " << (epoch + 1) << "/" << config_.num_epochs << " ===" << std::endl;
        train_data_.reset_cursor();
        float epoch_loss = 0.0f;
        int steps = 0;
        bool stop_early = false;

        while (true) {
            auto batch = train_data_.next_batch(config_.micro_batch_size, false);
            if (!batch.input_ids) break;

            float loss = train_step(batch);
            if (loss < 0.0f) {
                continue;
            }

            epoch_loss += loss;
            steps++;

            if (global_step_ % config_.logging_steps == 0) {
                float ppl = perplexity_from_loss(loss);
                std::cout << "[Step " << global_step_ << "] Loss=" << loss
                          << " PPL=" << ppl << " LR=" << get_lr(global_step_) << std::endl;
                std::cout << std::fixed << std::setprecision(3)
                          << "RSS(pre/fwd/bwd/opt/post)="
                          << last_rss_pre_mb_ << "/" << last_rss_fwd_mb_ << "/" << last_rss_bwd_mb_
                          << "/" << last_rss_opt_mb_ << "/" << last_rss_post_mb_
                          << " MB | step_max=" << last_step_max_rss_mb_
                          << " MB | peak=" << peak_rss_mb_ << " MB" << std::endl;
            }

            if (config_.eval_steps > 0 && global_step_ % config_.eval_steps == 0) {
                evaluate();
            }

            if (config_.save_every > 0 && global_step_ % config_.save_every == 0) {
                const auto ckpt_path = make_checkpoint_path(config_.output_dir, global_step_);
                std::cout << "[Checkpoint] saving to " << ckpt_path << std::endl;
                save_lora(ckpt_path);
            }

            if (config_.max_steps > 0 && global_step_ >= config_.max_steps) {
                stop_early = true;
                break;
            }

            // Energy-friendly dynamic sleep
            int sleep_ms = power_monitor_.suggest_sleep_ms(global_step_);
            if (sleep_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
        }

        float avg_loss = (steps > 0) ? epoch_loss / steps : 0.0f;
        std::cout << "Epoch " << (epoch + 1) << " avg loss: " << avg_loss << std::endl;
        MemoryManager::instance().cleanup_dead_references();
        MemoryManager::instance().clear_unused_memory();

        if (stop_early) {
            std::cout << "[GemmaTrainer] Reached max_steps=" << config_.max_steps << ", stopping early." << std::endl;
            break;
        }
    }

    MemoryManager::instance().force_cleanup();
    std::cout << std::fixed << std::setprecision(3)
              << "[RSSSummary] Gemma MaxRSS(train) = " << peak_rss_mb_ << " MB" << std::endl;
}

void GemmaLoRATrainer::save_lora(const std::string& path) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    injector_.save_lora_safetensors(path);
}

}  // namespace ops
