#include "inspectflow/detection_store.hpp"
#include "inspectflow/detection_record.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {
    try {
        if (argc != 2) {
            throw std::runtime_error(
                "Usage: test_detection_store <database_path>"
            );
        }

        DetectionStore store(argv[1]);

        DetectionRecord input;
        input.created_at = "2026-11-04T00:00:00Z";
        input.request_id = "test-request";
        input.model_name = "TEST_MODEL_NOT_REAL";
        input.model_version = 1;
        input.pred_score = 0.75;
        input.pred_label = true;
        input.inference_ms = 12.5;

        const auto id = store.insert(input);
        const auto output = store.find_by_id(5);

        if (!output.has_value()) {
            throw std::runtime_error("Inserted record not found");
        }

        const auto& actual = output->record;

        if (output->detection_id != id ||
            actual.created_at != input.created_at ||
            actual.request_id != input.request_id ||
            actual.model_name != input.model_name ||
            actual.model_version != input.model_version ||
            actual.pred_label != input.pred_label ||
            std::abs(actual.pred_score - input.pred_score) > 1e-9 ||
            std::abs(actual.inference_ms - input.inference_ms) > 1e-9) {
            throw std::runtime_error("Round-trip data mismatch");
        }

        std::cout << "PASS: inserted and read detection "
                  << id << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}