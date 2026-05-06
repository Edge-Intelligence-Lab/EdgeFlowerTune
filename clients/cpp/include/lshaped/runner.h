#pragma once

#include "lshaped/client_config.h"

namespace lshaped {

int RunClient(const ClientOptions& options);
bool IsSplitRunMode(const ClientOptions& options);

}  // namespace lshaped
