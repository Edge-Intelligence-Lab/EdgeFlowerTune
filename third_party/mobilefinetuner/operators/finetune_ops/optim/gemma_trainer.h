#pragma once

#include "../core/tensor.h"
#include "../graph/gemma_model.h"
#include "../graph/gemma_lora_injector.h"
#include "adam.h"
#include "../../opt_ops/energy/power_monitor.h"
#include <memory>
#include <string>
#include <vector>

namespace ops {
class WikiText2Dataset;
struct Batch;
}

namespace ops {

struct GemmaTrainerConfig {
    float learning_rate = 2e-4f;
    float weight_decay = 0.0f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1e-8f;

    int num_epochs = 3;
    float max_grad_norm = 1.0f;

    std::string lr_scheduler = "linear";
    float warmup_ratio = 0.03f;

    int logging_steps = 10;
    int eval_steps = 100;
    std::string output_dir = "./gemma_lora_ckpt";
    int save_every = 0;
    int max_steps = -1;
    int micro_batch_size = 1;
    int grad_accum_steps = 1;
    float loss_scale = 1.0f;
    float fedprox_mu = 0.0f;
    bool dump_embedding = false;
    int dump_embedding_step = 1;
    std::string dump_embedding_dir = "./debug";

    // Energy-aware throttling
    int pm_interval = 0;          // every K steps; 0 = off
    float pm_batt_thresh = 20.0f;
    float pm_temp_thresh = 42.0f;
    float pm_fb_high = 2.0f, pm_fb_low = 0.5f;
    float pm_ft_high = 2.0f, pm_ft_low = 0.5f;
    float pm_manual_batt = 100.0f;
    float pm_manual_temp = 30.0f;
    bool pm_enable_batt = true;
    bool pm_enable_temp = true;
    std::string pm_schedule;
};

class GemmaLoRATrainer {
public:
    GemmaLoRATrainer(GemmaModel& model,
                     GemmaLoraInjector& injector,
                     WikiText2Dataset& train_data,
                     WikiText2Dataset& eval_data,
                     const GemmaTrainerConfig& config);

    void train();
    float train_step(const Batch& batch);
    float evaluate();
    void save_lora(const std::string& path);
    float last_prox_term() const { return last_prox_term_; }

private:
    GemmaModel& model_;
    GemmaLoraInjector& injector_;
    WikiText2Dataset& train_data_;
    WikiText2Dataset& eval_data_;
    GemmaTrainerConfig config_;

    std::unique_ptr<Adam> optimizer_;
    energy::PowerMonitor power_monitor_;
    int global_step_;
    int accum_counter_ = 0;
    float accum_loss_ = 0.0f;
    int64_t micro_step_counter_ = 0;
    bool embedding_dump_scheduled_ = false;
    std::vector<std::vector<float>> proximal_reference_;
    float last_prox_term_ = 0.0f;
    double last_rss_pre_mb_ = 0.0;
    double last_rss_fwd_mb_ = 0.0;
    double last_rss_bwd_mb_ = 0.0;
    double last_rss_opt_mb_ = 0.0;
    double last_rss_post_mb_ = 0.0;
    double last_step_max_rss_mb_ = 0.0;
    double peak_rss_mb_ = 0.0;

    float get_lr(int step);
    float apply_fedprox_gradients();
    void clip_gradients();
};

}  // namespace ops
