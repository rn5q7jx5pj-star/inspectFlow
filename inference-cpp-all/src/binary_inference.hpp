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
#include <opencv2/core/version.hpp>

namespace fs = std::filesystem;

struct InferenceArgs {
    fs::path model;
    fs::path image;
    fs::path input_bin;
    fs::path output;
};
/*parse_inference_args用于解析命令行的输入
逻辑是用target找出对应InferenceArgs结构体的地址 需要检查target是否为空；若为空才能进行赋值，否则说明之前赋过值
然后给target所指向的地址复制
*/
inline InferenceArgs parse_inference_args(int argc, char* argv[]) {
    InferenceArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        fs::path* target = nullptr;
        if (key == "--model") target = &args.model;
        else if (key == "--image") target = &args.image;
        else if (key == "--input-bin") target = &args.input_bin;
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
    //检查是什么模式
    const bool has_image = !args.image.empty();
    const bool has_binary = !args.input_bin.empty();
    const bool has_output = !args.output.empty();

    // 两种输入不能同时出现。
    if (has_image && has_binary) {
        throw std::invalid_argument(
            "--image and --input-bin are mutually exclusive"
        );
    }

    const bool has_input = has_image || has_binary;

    // 推理输入与输出目录必须配套。
    if (has_input && !has_output) {
        throw std::invalid_argument(
            "--output-dir is required when an input is provided"
        );
    }

    if (!has_input && has_output) {
        throw std::invalid_argument(
            "--output-dir requires --image or --input-bin"
        );
    }
    return args;
}

/*check_binary_platform
它检查当前机器的数据表示方式，是否符合我们约定的二进制文件协议。它不检查文件内容本身。
先检查浮点数格式-检查字节序（用1来测试大端00/01还是小端01/00
*reinterpret_cast<const unsigned char*>(&one)表示取第一个字节
*/
inline void check_binary_platform() {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "IEEE 754 float32 is required");
    const std::uint16_t one = 1;
    if (*reinterpret_cast<const unsigned char*>(&one) != 1) {
        throw std::runtime_error("This implementation requires a little-endian host");
    }
}

/*read_image用于在image输入模式中读取图片并进行预处理
包含读图-转RGB-resize大小-归一化-写到vector<float>中（由HWC转为CHW
*/
inline std::vector<float> read_image(const std::string& path) {
    cv::Mat image = cv::imread(path,cv::IMREAD_COLOR);

    if (image.empty()) {
        throw std::runtime_error(
            "Cannot read or decode image: " + path
        );
    }
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

/*read_input_bin用于读取二进制文件
*/
inline std::vector<float> read_input_bin(const std::string& path) {
    constexpr std::size_t element_count = 1 * 3 * 256 * 256;
    constexpr std::size_t expected_bytes =
        element_count * sizeof(float);
    // 1. 以二进制模式打开文件，并定位到文件末尾。
    std::ifstream file(
        path,
        std::ios::binary | std::ios::ate
    );
    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open input file"
        );
    }
    // 2. 获取文件大小。
    const auto end_position = file.tellg();
    if (end_position == std::ifstream::pos_type(-1)) {
        throw std::runtime_error("Cannot determine file size");
    }
    if (static_cast<std::streamoff>(end_position)
        != static_cast<std::streamoff>(expected_bytes)) {
        throw std::runtime_error(
            "Input file must contain exactly 786432 bytes"
        );
    }
    // 3. 回到文件开头。
    file.seekg(0, std::ios::beg);
    if (!file) {
        throw std::runtime_error("Cannot seek to file beginning");
    }
    // 4. 创建足够大的float数组。
    std::vector<float> data(element_count);
    // 5. 将文件字节直接读入数组内存。
    //data()是 std::vector 自带的成员函数，用于获取内部元素数组的首地址
    //data.data()返回的是 float*，所以需要将同一个地址转换成 char*，让 read()按字节写入
    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(expected_bytes)
    );
    if (!file) {
        throw std::runtime_error("Failed to read complete input");
    }
    // 6. 检查数值是否符合模型输入约定。
    for (float value : data) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Input contains NaN or Inf");
        }

        if (value < 0.0f || value > 1.0f) {
            throw std::runtime_error("Input must be within [0, 1]");
        }
    }
    return data;
}

/*require_tensor用于检查输出和预期是否相同
*/
inline void require_tensor(const Ort::Value& value, ONNXTensorElementDataType type,
                           const std::vector<int64_t>& shape, const char* name) {
    if (!value.IsTensor()) throw std::runtime_error(std::string(name) + " is not a tensor");
    const auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != type || info.GetShape() != shape) {
        throw std::runtime_error(std::string(name) + " has an unexpected dtype or shape");
    }
}

/*json_string用于把普通C++字符串转换成可以安全写入JSON的字符串文本，包括外层双引号和内部转义
比如我们输入\实际上要加转义\\才能表示一个\；这里就是实现这个功能 如果ch == '"' || ch == '\\' 则多加一个\表示转义
*/
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

//用于写入热力图和二值图文件
inline void write_bytes(const fs::path& path, const void* data, std::size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    stream.close();
}

//用于保存预处理后的输入数组
inline void save_input_tensor(
    const std::vector<float>& data,
    const fs::path& output_dir
) {
    if (!fs::is_directory(output_dir)) {
        throw std::runtime_error(
            "Output directory does not exist: " + output_dir.string()
        );
    }

    const auto input_path = output_dir / "input.f32";
    const auto metadata_path = output_dir / "input_metadata.json";

    if (fs::exists(input_path) || fs::exists(metadata_path)) {
        throw std::runtime_error(
            "Input tensor or metadata already exists"
        );
    }

    // 保存实际传给模型的float数组，不重新预处理。
    write_bytes(
        input_path,
        data.data(),
        data.size() * sizeof(float)
    );

    const auto bounds = std::minmax_element(
        data.begin(),
        data.end()
    );

    std::ostringstream metadata;
    metadata.imbue(std::locale::classic());

    metadata
        << std::setprecision(17)
        << "{\n"
        << "  \"file\": \"input.f32\",\n"
        << "  \"shape\": [1, 3, 256, 256],\n"
        << "  \"dtype\": \"float32\",\n"
        << "  \"byte_order\": \"little\",\n"
        << "  \"layout\": \"NCHW\",\n"
        << "  \"channels\": \"RGB\",\n"
        << "  \"value_range\": [0, 1],\n"
        << "  \"actual_min\": " << *bounds.first << ",\n"
        << "  \"actual_max\": " << *bounds.second << ",\n"
        << "  \"byte_count\": "
        << data.size() * sizeof(float) << ",\n"
        << "  \"read_mode\": \"imread IMREAD_COLOR -> BGR2RGB\",\n"
        << "  \"resize\": \"OpenCV INTER_LINEAR\",\n"
        << "  \"library\": \"opencv\",\n"
        << "  \"library_version\": "
        << json_string(CV_VERSION) << ",\n"
        << "  \"normalization\": "
        << "\"convertTo CV_32F, scale=1/255; "
        << "no external ImageNet normalization\"\n"
        << "}\n";

    const std::string text = metadata.str();

    write_bytes(
        metadata_path,
        text.data(),
        text.size()
    );
}

//主函数 实现预处理-模型推理-数据存储
inline void run_binary_inference(Ort::Session& session,
                                 Ort::AllocatorWithDefaultOptions& allocator,
                                 const InferenceArgs& args) {
    check_binary_platform();
    const auto output_path = fs::absolute(args.output);
    if (fs::exists(output_path)) throw std::runtime_error("Output directory already exists; choose a new directory");
    fs::create_directories(output_path.parent_path());
    if (!fs::create_directory(output_path)) throw std::runtime_error("Output directory already exists");

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
    std::vector<float> data; 
    if(!args.image.empty()){
        data = read_image(fs::absolute(args.image));
        save_input_tensor(data, output_path);
    }else if(!args.input_bin.empty()){
        data = read_input_bin(fs::absolute(args.input_bin));
    }else return;

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
           << ",\n  \"input_path\": " << json_string(!args.image.empty() ? fs::absolute(args.image) : fs::absolute(args.input_bin))
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

    //fs::create_directories(output_path.parent_path());往上移了
    //if (!fs::create_directory(output_path)) throw std::runtime_error("Output directory already exists");
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
