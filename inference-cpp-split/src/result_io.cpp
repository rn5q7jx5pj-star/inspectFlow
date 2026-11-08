#include <cstdint>
#include <onnxruntime_cxx_api.h>

#include "inspectflow/inference_engine.hpp"
#include "inspectflow/result.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

// JSON 字符串必须转义引号、反斜杠和控制字符，路径不能直接放进双引号里。
std::string json_string(const std::string& value) {
    std::ostringstream escaped;
    escaped << '"';
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') {
            escaped << '\\' << static_cast<char>(ch);
        } else if (ch < 0x20) {
            escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(ch) << std::dec;
        } else {
            escaped << static_cast<char>(ch);
        }
    }
    escaped << '"';
    return escaped.str();
}

}  // namespace

void Result::write_bytes(const std::filesystem::path& path,
                         const void* data, std::size_t bytes) {
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("Output file already exists: " + path.string());
    }
    std::ofstream stream(path, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    stream.close();
}

void Result::write_json(const std::filesystem::path& output_dir,
                        const Prediction& prediction) {
    constexpr std::size_t pixels = 256 * 256;
    if (!std::filesystem::is_directory(output_dir)) {
        throw std::runtime_error("Output directory does not exist: " + output_dir.string());
    }
    if (prediction.anomaly_map.size() != pixels || prediction.mask.size() != pixels) {
        throw std::invalid_argument("Prediction maps must each contain 256*256 elements");
    }
    if (!std::isfinite(prediction.score) || !std::isfinite(prediction.run_ms) ||
        prediction.run_ms < 0) {
        throw std::invalid_argument("Prediction contains an invalid score or duration");
    }

    std::size_t anomaly_pixels = 0;
    for (std::size_t i = 0; i < pixels; ++i) {
        if (!std::isfinite(prediction.anomaly_map[i])) {
            throw std::invalid_argument("Prediction anomaly_map contains NaN or Inf");
        }
        if (prediction.mask[i] > 1) {
            throw std::invalid_argument("Prediction mask must contain only 0 or 1");
        }
        anomaly_pixels += prediction.mask[i];
    }
    const auto bounds = std::minmax_element(
        prediction.anomaly_map.begin(), prediction.anomaly_map.end());

    std::ostringstream report;
    report.imbue(std::locale::classic());
    report << std::setprecision(17) << std::boolalpha
           << "{\n  \"pred_score\": " << prediction.score
           << ",\n  \"pred_label\": " << prediction.label
           << ",\n  \"anomaly_map_shape\": [1, 1, 256, 256]"
           << ",\n  \"pred_mask_shape\": [1, 1, 256, 256]"
           << ",\n  \"anomaly_map_min\": " << *bounds.first
           << ",\n  \"anomaly_map_max\": " << *bounds.second
           << ",\n  \"predicted_anomaly_pixels\": " << anomaly_pixels
           << ",\n  \"session_run_ms\": " << prediction.run_ms
           << "\n}\n";

    const auto text = report.str();
    write_bytes(output_dir / "prediction.json", text.data(), text.size());
}

cv::Mat Result::create_heatmap(const Prediction& prediction) {
    constexpr int size = 256;
    constexpr std::size_t pixels = size * size;
    if (prediction.anomaly_map.size() != pixels) {
        throw std::invalid_argument("Prediction anomaly_map must contain 256*256 elements");
    }
    cv::Mat float_map(size, size, CV_32FC1);
    for (std::size_t i = 0; i < pixels; ++i) {
        const float value = prediction.anomaly_map[i];
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Prediction anomaly_map contains NaN or Inf");
        }
        float_map.ptr<float>()[i] = value;
    }
    // 逐图归一化只用于展示；不同图片的颜色深浅不能直接比较异常程度。
    cv::Mat gray_map;
    cv::normalize(float_map, gray_map, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat heatmap;
    cv::applyColorMap(gray_map, heatmap, cv::COLORMAP_JET);
    return heatmap;
}

cv::Mat Result::create_overlay(const cv::Mat& original_bgr, const cv::Mat& heatmap) {
    if (original_bgr.empty() || original_bgr.type() != CV_8UC3) {
        throw std::invalid_argument("Original image must be a non-empty 8-bit BGR image");
    }
    if (heatmap.size() != cv::Size(256, 256) || heatmap.type() != CV_8UC3) {
        throw std::invalid_argument("Heatmap must be a 256x256 8-bit BGR image");
    }
    cv::Mat resized, overlay;
    cv::resize(original_bgr, resized, heatmap.size(), 0, 0, cv::INTER_LINEAR);
    cv::addWeighted(resized, 0.6, heatmap, 0.4, 0.0, overlay);
    return overlay;
}

void Result::write_images(const std::filesystem::path& output_dir,
                          const Prediction& prediction,
                          const cv::Mat& original_bgr) {
    constexpr std::size_t pixels = 256 * 256;
    if (!std::filesystem::is_directory(output_dir)) {
        throw std::runtime_error("Output directory does not exist: " + output_dir.string());
    }
    if (prediction.mask.size() != pixels) {
        throw std::invalid_argument("Prediction mask must contain 256*256 elements");
    }
    const auto heatmap_path = output_dir / "heatmap.png";
    const auto mask_path = output_dir / "mask.png";
    const auto overlay_path = output_dir / "overlay.png";
    if (std::filesystem::exists(heatmap_path) || std::filesystem::exists(mask_path) ||
        (!original_bgr.empty() && std::filesystem::exists(overlay_path))) {
        throw std::runtime_error("Image output file already exists");
    }
    cv::Mat heatmap = create_heatmap(prediction);
    cv::Mat mask_image(256, 256, CV_8UC1);
    for (std::size_t i = 0; i < pixels; ++i) {
        if (prediction.mask[i] > 1) {
            throw std::invalid_argument("Prediction mask must contain only 0 or 1");
        }
        mask_image.data[i] = prediction.mask[i] ? 255 : 0;
    }
    cv::Mat overlay;
    if (!original_bgr.empty()) overlay = create_overlay(original_bgr, heatmap);

    if (!cv::imwrite(heatmap_path.string(), heatmap)) {
        throw std::runtime_error("Failed to save heatmap: " + heatmap_path.string());
    }
    if (!cv::imwrite(mask_path.string(), mask_image)) {
        throw std::runtime_error("Failed to save mask: " + mask_path.string());
    }
    if (!overlay.empty() && !cv::imwrite(overlay_path.string(), overlay)) {
        throw std::runtime_error("Failed to save overlay: " + overlay_path.string());
    }
}

void Result::write_run_info_json(const std::filesystem::path& output_dir,
                                 const RunInfo& info) {
    if (!std::filesystem::is_directory(output_dir)) {
        throw std::runtime_error("Output directory does not exist: " + output_dir.string());
    }
    if (info.image_count != info.success_count + info.failure_count) {
        throw std::invalid_argument("RunInfo image counts are inconsistent");
    }
    if (!std::isfinite(info.model_load_ms) || info.model_load_ms < 0 ||
        !std::isfinite(info.session_run_total_ms) || info.session_run_total_ms < 0 ||
        !std::isfinite(info.batch_wall_ms) || info.batch_wall_ms < 0) {
        throw std::invalid_argument("RunInfo contains an invalid duration");
    }

    std::ostringstream report;
    report.imbue(std::locale::classic());
    report << std::setprecision(17)
           << "{\n  \"model\": {\n"
           << "    \"path\": " << json_string(info.model_path.string()) << ",\n"
           << "    \"file_name\": " << json_string(info.model_path.filename().string()) << ",\n"
           << "    \"file_size_bytes\": " << std::filesystem::file_size(info.model_path) << ",\n"
           << "    \"onnx_metadata_version\": ";
    if (info.model_version == std::numeric_limits<std::int64_t>::max()) {
        report << "null"; // ORT 对此模型返回占位值，不当作有意义的版本号。
    } else {
        report << info.model_version;
    }
    report << "\n"
           << "  },\n"
           << "  \"image_list_path\": " << json_string(info.image_list_path.string()) << ",\n"
           << "  \"preprocessing\": {\n"
           << "    \"input_shape\": [1, 3, 256, 256],\n"
           << "    \"input_dtype\": \"float32\",\n"
           << "    \"layout\": \"NCHW\",\n"
           << "    \"color_order\": \"RGB\",\n"
           << "    \"image_read\": \"OpenCV IMREAD_COLOR (BGR)\",\n"
           << "    \"resize\": \"256x256 INTER_LINEAR\",\n"
           << "    \"normalization\": \"divide by 255; no ImageNet normalization\"\n"
           << "  },\n"
           << "  \"counts\": {\n"
           << "    \"images\": " << info.image_count << ",\n"
           << "    \"success\": " << info.success_count << ",\n"
           << "    \"failure\": " << info.failure_count << "\n"
           << "  },\n"
           << "  \"versions\": {\n"
           << "    \"onnxruntime\": " << json_string(OrtGetApiBase()->GetVersionString()) << ",\n"
           << "    \"opencv\": " << json_string(CV_VERSION) << "\n"
           << "  },\n"
           << "  \"timing\": {\n"
           << "    \"model_load_ms\": " << info.model_load_ms << ",\n"
           << "    \"session_run_count\": " << info.session_run_count << ",\n"
           << "    \"session_run_total_ms\": " << info.session_run_total_ms << ",\n"
           << "    \"session_run_mean_ms\": ";
    if (info.session_run_count == 0) {
        report << "null";
    } else {
        report << info.session_run_total_ms / info.session_run_count;
    }
    report << ",\n    \"batch_wall_ms\": " << info.batch_wall_ms << ",\n"
           << "    \"scope\": \"Batch wall time includes list reading, model loading, preprocessing, Session::Run, saving images and predictions.csv; excludes writing run_info.json. Session::Run time excludes preprocessing and file IO.\"\n"
           << "  }\n}\n";

    const auto text = report.str();
    write_bytes(output_dir / "run_info.json", text.data(), text.size());
}

void Result::write_benchmark_json(const std::filesystem::path& output_dir,
                                  double model_load_ms,
                                  double preprocess_ms,
                                  double first_run_ms,
                                  double median_ms,
                                  double p95_ms) {
    if (!std::filesystem::is_directory(output_dir)) {
        throw std::runtime_error(
            "Output directory does not exist: " + output_dir.string());
    }

    const double durations[] = {
        model_load_ms, preprocess_ms, first_run_ms, median_ms, p95_ms
    };
    for (double duration : durations) {
        if (!std::isfinite(duration) || duration < 0) {
            throw std::invalid_argument("Invalid benchmark duration");
        }
    }

    std::ostringstream report;
    report.imbue(std::locale::classic());
    report << std::setprecision(17)
           << "{\n"
           << "  \"model_load_ms\": " << model_load_ms << ",\n"
           << "  \"preprocess_ms\": " << preprocess_ms << ",\n"
           << "  \"first_run_ms\": " << first_run_ms << ",\n"
           << "  \"warmup_runs\": 5,\n"
           << "  \"measured_runs\": 50,\n"
           << "  \"median_run_ms\": " << median_ms << ",\n"
           << "  \"p95_run_ms\": " << p95_ms << ",\n"
           << "  \"p95_method\": \"nearest_rank\"\n"
           << "}\n";

    const auto text = report.str();
    write_bytes(output_dir / "benchmark.json", text.data(), text.size());
}
