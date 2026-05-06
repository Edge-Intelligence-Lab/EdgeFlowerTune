#include "lshaped/runner.h"

#include <memory>

#include "lshaped/federated_trainer.h"
#include "lshaped/flower_legacy_client.h"
#include "lshaped/mmlu_dataset.h"
#include "lshaped/prefix_encoder.h"
#include "lshaped/split_flower_client.h"

namespace lshaped {

bool IsSplitRunMode(const ClientOptions& options) {
    return options.run_mode == "split";
}

bool IsWikiTextRawMode(const ClientOptions& options) {
    return options.dataset_format == "wikitext_raw";
}

int RunClient(const ClientOptions& options) {
    if (IsSplitRunMode(options)) {
        ClientShardDataset dataset;
        if (!IsWikiTextRawMode(options)) {
            dataset = ClientShardDataset::FromOptions(options);
        }
        std::unique_ptr<PrefixEncoder> encoder = CreatePrefixEncoder(options);
        SplitFlowerClient client(options, std::move(dataset), std::move(encoder));
        return client.Run();
    }
    if (IsWikiTextRawMode(options)) {
        std::unique_ptr<FederatedTrainer> trainer = CreateFederatedTrainer(options, ClientShardDataset{});
        FlowerLegacyClient client(options, std::move(trainer));
        return client.Run();
    }
    ClientShardDataset dataset = ClientShardDataset::FromOptions(options);
    std::unique_ptr<FederatedTrainer> trainer = CreateFederatedTrainer(options, std::move(dataset));
    FlowerLegacyClient client(options, std::move(trainer));
    return client.Run();
}

}  // namespace lshaped
