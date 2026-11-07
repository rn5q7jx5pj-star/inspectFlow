#include "inspectflow/detection_store.hpp"

#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Expected {
    std::int64_t id;
    DetectionRecord record;
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Usage: test_detection_store_concurrency <new-test-database-path>");
        }
        const std::filesystem::path path(argv[1]);
        if (std::filesystem::exists(path)) {
            throw std::runtime_error("Test database already exists; choose a new path");
        }
        constexpr int thread_count = 4;
        constexpr int per_thread = 10;
        DetectionStore store(path);
        std::vector<Expected> expected;
        std::vector<std::string> errors;
        std::mutex results_mutex;
        std::atomic<int> ready{0};
        std::atomic<bool> start{false};
        std::vector<std::thread> threads;

        for (int t = 0; t < thread_count; ++t) {
            threads.emplace_back([&, t] {
                ++ready;
                while (!start.load()) std::this_thread::yield();
                for (int i = 0; i < per_thread; ++i) {
                    DetectionRecord record;
                    record.request_id = "test-t" + std::to_string(t) + "-" + std::to_string(i);
                    record.model_name = "TEST_MODEL_NOT_REAL";
                    record.model_version = t;
                    record.pred_score = t * 100.0 + i;
                    record.pred_label = i % 2 != 0;
                    record.inference_ms = i + 0.5;
                    try {
                        const auto id = store.insert(record);
                        std::lock_guard<std::mutex> guard(results_mutex);
                        expected.push_back({id, record});
                    } catch (const std::exception& e) {
                        std::lock_guard<std::mutex> guard(results_mutex);
                        errors.push_back(record.request_id + ": " + e.what());
                    }
                }
            });
        }
        while (ready.load() != thread_count) std::this_thread::yield();
        start = true;
        for (auto& thread : threads) thread.join();

        for (const auto& error : errors) std::cerr << error << '\n';
        if (!errors.empty() || expected.size() != thread_count * per_thread)
            throw std::runtime_error("Some inserts failed");
        std::set<std::int64_t> ids;
        for (const auto& item : expected) {
            if (!ids.insert(item.id).second) throw std::runtime_error("Duplicate ID");
            const auto found = store.find_by_id(item.id);
            if (!found) throw std::runtime_error("Inserted record missing");
            const auto& actual = found->record;
            if (found->detection_id != item.id || actual.request_id != item.record.request_id ||
                actual.model_name != item.record.model_name || actual.model_version != item.record.model_version ||
                actual.pred_label != item.record.pred_label ||
                std::abs(actual.pred_score - item.record.pred_score) > 1e-9 ||
                std::abs(actual.inference_ms - item.record.inference_ms) > 1e-9)
                throw std::runtime_error("ID/record mismatch: " + std::to_string(item.id));
        }
        std::cout << "PASS: " << expected.size() << " concurrent inserts and ID mappings verified\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
