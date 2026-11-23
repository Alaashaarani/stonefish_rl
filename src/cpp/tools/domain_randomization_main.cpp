#include "DomainRandomization.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
int main(int argc, char** argv) {
    if (argc != 6 && !(argc == 7 && std::string(argv[6]) == "--report-only")) {
        std::cerr << "Usage: StonefishDomainRandomization SCENE RESOURCES TARGETS_YAML OUTPUT_DIR REPORT_YAML [--report-only]\n";
        return 2;
    }
    try {
        auto targets = YAML::LoadFile(argv[3])["targets"];
        auto report = GenerateRandomizedScenario(argv[1], argv[2], argv[4], targets, argc == 6);
        std::ofstream output(argv[5]); output << report;
        if (!output) throw std::runtime_error("cannot write report");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
