#pragma once

#include <cstdint>
#include <optional>
#include <string>

// 准备写入数据库的检测数据
struct DetectionRecord {
    std::string created_at;
    std::string request_id;

    std::string model_name;
    std::optional<std::int64_t> model_version;

    double pred_score = 0.0;
    bool pred_label = false;
    double inference_ms = 0.0;
};

// 从数据库中查出的检测记录
struct StoredDetection {
    std::int64_t detection_id = 0;
    DetectionRecord record;
};




