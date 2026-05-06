#include <exception>
#include <iostream>

#include "lshaped/client_config.h"
#include "lshaped/runner.h"

int main(int argc, char** argv) {
    try {
        const lshaped::ClientOptions options = lshaped::ParseClientOptions(argc, argv);
        lshaped::PrintClientOptions(options);
        return lshaped::RunClient(options);
    } catch (const std::exception& ex) {
        std::cerr << "[fatal] " << ex.what() << "\n";
        return 1;
    }
}
