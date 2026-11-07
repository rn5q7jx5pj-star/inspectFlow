#pragma once

#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

struct InferenceArgs {
    fs::path model;
    fs::path input;
    fs::path output;
};

inline InferenceArgs parse_inference_args(int argc, char* argv[]) {
    InferenceArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        fs::path* target = nullptr;
        if (key == "--model") target = &args.model;
        else if (key == "--image") target = &args.input;
        else if (key == "--output-dir") target = &args.output;
        else throw std::invalid_argument("Unknown option: " + key);
        if (!target->empty()) throw std::invalid_argument("Duplicate option: " + key);
        if (++i >= argc || std::string(argv[i]).empty() ||
            std::string(argv[i]).rfind("--", 0) == 0) {
            throw std::invalid_argument("Missing value for " + key);
        }
        *target = argv[i];
    }
    if (args.model.empty()) throw std::invalid_argument("--model is required");
    if (args.input.empty() != args.output.empty()) {
        throw std::invalid_argument("--image and --output-dir must be used together");
    }
    return args;
}

inline void check_binary_platform() {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "IEEE 754 float32 is required");
    const std::uint16_t one = 1;
    if (*reinterpret_cast<const unsigned char*>(&one) != 1) {
        throw std::runtime_error("This implementation requires a little-endian host");
    }
}

inline std::vector<float> read_input(const std::string& path) {
    cv::Mat image = cv::imread(path,cv::IMREAD_COLOR);

    cv::Mat rgb;
    cv::cvtColor(image,rgb,cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb,resized,cv::Size(256,256));
    cv::Mat float_image;
    resized.convertTo(float_image, CV_32F, 1.0 / 255.0);

    const int height = float_image.rows;
    const int width = float_image.cols;
    std::vector<float> input(3 * height * width);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            const cv::Vec3f pixel = float_image.at<cv::Vec3f>(i,j);
            for(int k = 0; k < 3; k++){
                input[k*height*width+width*i+j] = pixel[k];
            }
        }
    }
    return input;
}

inline void require_tensor(const Ort::Value& value, ONNXTensorElementDataType type,
                           const std::vector<int64_t>& shape, const char* name) {
    if (!value.IsTensor()) throw std::runtime_error(std::string(name) + " is not a tensor");
    const auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != type || info.GetShape() != shape) {
        throw std::runtime_error(std::string(name) + " has an unexpected dtype or shape");
    }
}

inline std::string json_string(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : text) {
        if (ch == '"' || ch == '\\') out << '\\' << static_cast<char>(ch);
        else if (ch < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
        else out << static_cast<char>(ch);
    }
    out << '"';
    return out.str();
}

inline void write_bytes(const fs::path& path, const void* data, std::size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    stream.close();
}

inline void run_binary_inference(Ort::Session& session,
                                 Ort::AllocatorWithDefaultOptions& allocator,
                                 const InferenceArgs& args) {
    check_binary_platform();
    const auto input_path = fs::absolute(args.input);
    const auto output_path = fs::absolute(args.output);
    if (fs::exists(output_path)) throw std::runtime_error("Output directory already exists; choose a new directory");

    // 固定输入协议。type_info拥有元数据，input_info只在其有效期内使用。
    if (session.GetInputCount() != 1) throw std::runtime_error("Expected one model input");
    const auto type_info = session.GetInputTypeInfo(0);
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) throw std::runtime_error("Expected tensor input");
    const auto input_info = type_info.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape{1, 3, 256, 256};
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        input_info.GetShape() != shape) {
        throw std::runtime_error("Model input must be float32 [1,3,256,256]");
    }

    // Tensor借用vector内存：直到Run结束前，不能销毁、resize或重新分配data。
    auto data = read_input(args.input);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        memory, data.data(), data.size(), shape.data(), shape.size());
    const auto input_name = session.GetInputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};

    // 显式请求名称，返回顺序由这里决定，不依赖模型声明顺序。
    const std::array<const char*, 4> names{
        "pred_score", "pred_label", "anomaly_map", "pred_mask"};
    for (const auto* required : names) {
        bool found = false;
        for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
            const auto name = session.GetOutputNameAllocated(i, allocator);
            if (std::string(name.get()) == required) found = true;
        }
        if (!found) throw std::runtime_error(std::string("Missing model output: ") + required);
    }
    const auto start = std::chrono::steady_clock::now();
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                               names.data(), names.size());
    const double run_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    require_tensor(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}, names[0]);
    require_tensor(outputs[1], ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, {1}, names[1]);
    require_tensor(outputs[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1,1,256,256}, names[2]);
    require_tensor(outputs[3], ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, {1,1,256,256}, names[3]);
    // 指针借用outputs拥有的内存；读取及写文件期间outputs必须保持存活。
    const float score = outputs[0].GetTensorData<float>()[0];
    const bool label = outputs[1].GetTensorData<bool>()[0];
    const float* map = outputs[2].GetTensorData<float>();
    const bool* mask = outputs[3].GetTensorData<bool>();
    constexpr std::size_t pixels = 256 * 256;
    if (!std::isfinite(score)) throw std::runtime_error("Non-finite pred_score");
    std::vector<std::uint8_t> mask_bytes(pixels);
    std::size_t anomaly_pixels = 0;
    for (std::size_t i = 0; i < pixels; ++i) {
        if (!std::isfinite(map[i])) throw std::runtime_error("Non-finite anomaly_map");
        // 文件约定用uint8 0/1，不依赖bool的内存表示，也不使用vector<bool>。
        mask_bytes[i] = mask[i] ? 1 : 0;
        anomaly_pixels += mask_bytes[i];
    }
    const auto bounds = std::minmax_element(map, map + pixels);
    std::ostringstream report;
    report.imbue(std::locale::classic());
    report << std::setprecision(17) << std::boolalpha
           << "{\n  \"model_path\": " << json_string(fs::absolute(args.model).string())
           << ",\n  \"input_path\": " << json_string(input_path.string())
           << ",\n  \"onnxruntime_version\": " << json_string(OrtGetApiBase()->GetVersionString())
           << ",\n  \"provider\": \"CPUExecutionProvider\","
           << "\n  \"input_shape\": [1,3,256,256],"
           << "\n  \"input_dtype\": \"float32\", \"layout\": \"NCHW\", \"channels\": \"RGB\","
           << "\n  \"byte_order\": \"little\","
           << "\n  \"pred_score\": " << score << ",\n  \"pred_label\": " << label
           << ",\n  \"anomaly_map_shape\": [1,1,256,256],"
           << "\n  \"anomaly_map_file\": \"anomaly_map.f32\","
           << "\n  \"pred_mask_file\": \"pred_mask.u8\", \"pred_mask_dtype\": \"uint8\","
           << "\n  \"anomaly_map_min\": " << *bounds.first
           << ",\n  \"anomaly_map_max\": " << *bounds.second
           << ",\n  \"predicted_anomaly_pixels\": " << anomaly_pixels
           << ",\n  \"session_run_ms\": " << run_ms
           << ",\n  \"timing_note\": \"First Session::Run only; excludes model loading, file IO and saving; not a steady-state benchmark\"\n}\n";

    fs::create_directories(output_path.parent_path());
    if (!fs::create_directory(output_path)) throw std::runtime_error("Output directory already exists");
    write_bytes(output_path / "anomaly_map.f32", map, pixels * sizeof(float));
    write_bytes(output_path / "pred_mask.u8", mask_bytes.data(), mask_bytes.size());
    // JSON最后写入：写文件失败时可能留下不完整目录，重试请换新目录。
    const auto text = report.str();
    write_bytes(output_path / "prediction.json", text.data(), text.size());
    std::cout << std::setprecision(9) << "pred_score: " << score
              << "\npred_label: " << std::boolalpha << label
              << "\nanomaly pixels: " << anomaly_pixels
              << "\nSession::Run: " << run_ms << " ms\nSaved: " << output_path << '\n';
}
