#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "inspectflow/preprocess.hpp"
#include "inspectflow/inference_engine.hpp"
#include "inspectflow/result.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;
// ONNX保存的是类型枚举，这里将常用类型转换成易读的名称。
const char* element_type_name(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "string";
        default: return "other";
    }
}

struct InferenceArgs {
    fs::path model;
    fs::path image;
    fs::path image_list;
    fs::path input_bin;
    fs::path output;
    bool benchmark = false;
};

struct CsvRow {
    fs::path image_path;  // 绝对路径
    bool success = false;
    float score = 0;
    bool label = false;
    double run_ms = 0;
    std::string error;
};

void print_tensor_info(const char* name, const Ort::TypeInfo& type_info) {
    std::cout << "  name: " << name << '\n';
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        std::cout << "  non-tensor ONNX type: "
                  << static_cast<int>(type_info.GetONNXType()) << "\n\n";
        return;
    }

    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    const auto element_type = tensor_info.GetElementType();
    const auto shape = tensor_info.GetShape();
    std::vector<const char*> symbols(shape.size(), nullptr);
    if (!symbols.empty()) {
        tensor_info.GetSymbolicDimensions(symbols.data(), symbols.size());
    }

    std::cout << "  dtype: " << element_type_name(element_type)
              << " (ONNX enum=" << static_cast<int>(element_type) << ")\n";
    std::cout << "  shape: [";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) std::cout << ", ";
        if (shape[i] >= 0) {
            std::cout << shape[i];
        } else if (symbols[i] != nullptr && symbols[i][0] != '\0') {
            std::cout << symbols[i] << " (dynamic)";
        } else {
            std::cout << "? (dynamic)";
        }
    }
    std::cout << "]\n\n";
}

/*print_usage为使用说明，如果是运行--help就打印这一段
*/
void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " --model <model.onnx>\n"
        << "  [--image <image.png> | --image-list <image-list.txt> | --input-bin <input.f32>]\n"
        << "  [--output-dir <new-directory>]\n"
        << "Inference requires --output-dir.\n";
}

/*parse_inference_args用于解析命令行的输入
逻辑是用target找出对应InferenceArgs结构体的地址 需要检查target是否为空；若为空才能进行赋值，否则说明之前赋过值
然后给target所指向的地址复制
*/
InferenceArgs parse_inference_args(int argc, char* argv[]) {
    InferenceArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--benchmark"){
            args.benchmark = true;
            continue;
        }
        fs::path* target = nullptr;
        if (key == "--model") target = &args.model;
        else if (key == "--image") target = &args.image;
        else if (key == "--image-list") target = &args.image_list;
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
    const bool has_image = !args.image.empty();
    const bool has_image_list = !args.image_list.empty();
    const bool has_binary = !args.input_bin.empty();
    const bool has_output = !args.output.empty();
    if (static_cast<int>(has_image) + static_cast<int>(has_image_list) +
        static_cast<int>(has_binary) > 1) {
        throw std::invalid_argument(
            "--image, --image-list and --input-bin are mutually exclusive"
        );
    }
    const bool has_input = has_image || has_binary || has_image_list;
    if (has_input && !has_output) {
        throw std::invalid_argument(
            "--output-dir is required when an input is provided"
        );
    }
    if (has_image_list && args.benchmark) {
        throw std::invalid_argument(
            "has_image_list and benchmark aren't together"
        );
    }
    if (!has_input && has_output) {
        throw std::invalid_argument(
            "--output-dir requires --image, --image-list or --input-bin"
        );
    }
    return args;
}

void print_model_info(const fs::path& model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InspectFlow");
    Ort::Session session(env, model_path.c_str(), Ort::SessionOptions{});
    Ort::AllocatorWithDefaultOptions allocator;
    std::cout << "Model loaded: " << model_path << "\n\n";
    std::cout << "Inputs: " << session.GetInputCount() << '\n';
    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
        const auto name = session.GetInputNameAllocated(i, allocator);
        const auto type_info = session.GetInputTypeInfo(i);
        print_tensor_info(name.get(), type_info);
    }

    std::cout << "Outputs: " << session.GetOutputCount() << '\n';
    for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
        const auto name = session.GetOutputNameAllocated(i, allocator);
        const auto type_info = session.GetOutputTypeInfo(i);
        print_tensor_info(name.get(), type_info);
    }
}

//读取txt列表输入
std::vector<fs::path> read_image_list(const fs::path& list_path) {
    std::ifstream file(list_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open image list: " + list_path.string());
    }

    std::vector<fs::path> images;
    std::set<std::string> seen;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(file, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line_number == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
            line.erase(0, 3); // 兼容带 UTF-8 BOM 的文本文件。
        }
        if (line.empty()) continue;

        fs::path image_path = fs::path(line).lexically_normal();
        if (!image_path.is_absolute()) {
            throw std::runtime_error(
                "Expected absolute path at line " + std::to_string(line_number));
        }
        if (!seen.insert(image_path.generic_string()).second) {
            throw std::runtime_error(
                "Duplicate image at line " + std::to_string(line_number));
        }
        images.push_back(image_path);
    }

    if (!file.eof()) throw std::runtime_error("Failed to read image list");
    return images;
}

// CSV 字段中若有逗号、引号或换行，需要加引号；原有引号要写成两个。
std::string csv_escape(const std::string& text) {
    if (text.find_first_of(",\"\r\n") == std::string::npos) return text;

    std::string escaped = "\"";
    for (char ch : text) {
        if (ch == '"') escaped += '"';
        escaped += ch;
    }
    escaped += '"';
    return escaped;
}

//写入csv文件
void write_predictions_csv(const fs::path& csv_path,
                           const std::vector<CsvRow>& rows) {
    if (fs::exists(csv_path)) {
        throw std::runtime_error("CSV already exists: " + csv_path.string());
    }
    std::ofstream file(csv_path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create CSV: " + csv_path.string());

    file << "image_path,status,pred_score,pred_label,run_ms,error\n";
    file << std::setprecision(9);

    for (const auto& row : rows) {
        file << csv_escape(row.image_path.generic_string()) << ','
             << (row.success ? "ok" : "error") << ',';

        if (row.success) {
            file << row.score << ','
                 << (row.label ? "true" : "false") << ','
                 << row.run_ms << ',';
        } else {
            file << ",,,"; // 分数、标签、耗时三列留空
        }
        file << csv_escape(row.error) << '\n';
    }

    file.close();
    if (!file) throw std::runtime_error("Failed to write CSV: " + csv_path.string());
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 2 && (std::string(argv[1]) == "--help" ||
                          std::string(argv[1]) == "-h")) {
            print_usage(argv[0]);
            return 0;
        }
        const auto args = parse_inference_args(argc, argv);
        const auto model_path = fs::absolute(args.model);
        std::cout << "ONNX Runtime version: "
                  << OrtGetApiBase()->GetVersionString() << '\n';

        if (args.image.empty() && args.input_bin.empty() && args.image_list.empty()) {
            print_model_info(model_path);
            return 0;
        }

        const auto output_path = fs::absolute(args.output);
        if (fs::exists(output_path)) {
            throw std::runtime_error("Output directory already exists: " + output_path.string());
        }

        using Clock = std::chrono::steady_clock;
        const auto batch_start = Clock::now();
        const bool is_batch = !args.image_list.empty();
        std::vector<fs::path> images;
        if (is_batch) {
            images = read_image_list(fs::absolute(args.image_list));
            if (images.empty()) {
                throw std::invalid_argument("Image list is empty");
            }
        } else if (!args.image.empty()) {
            images.push_back(fs::absolute(args.image));
        }

        // 一次加载模型；批量模式中的每张图片复用同一个 Session。
        const auto model_load_start = Clock::now();
        InferenceEngine engine(model_path);
        const double model_load_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - model_load_start).count();
        if (!fs::create_directory(output_path)) {
            throw std::runtime_error("Cannot create output directory: " + output_path.string());
        }
        Result result;
        //创建性能变量
        double image_load_ms;
        double first_run_ms;
        std::vector<double> fifty_run_ms(50,0.0);
        double median;
        double p95;
        if (!args.input_bin.empty()) {
            const auto image_load_start = Clock::now();
            auto data = PreProcess::read_input_bin(args.input_bin);
            image_load_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - image_load_start).count();

            auto prediction = engine.predict(data);
            first_run_ms = prediction.run_ms;
            
            PreProcess::save_input_tensor(data, output_path, false);
            result.write_images(output_path, prediction);
            result.write_json(output_path, prediction); // 最后写入，作为结果保存完成的标记。
            std::cout << "pred_score: " << prediction.score
                      << "\npred_label: " << std::boolalpha << prediction.label
                      << "\nSession::Run: " << prediction.run_ms << " ms"
                      << "\nSaved: " << output_path << '\n';
            if(args.benchmark){
                //预热
                int count = 5;
                while(count--){
                    prediction = engine.predict(data);
                }
                //性能测试
                count = 50;
                while(count--){
                    prediction = engine.predict(data);
                    fifty_run_ms[49-count] = prediction.run_ms;
                }
                sort(fifty_run_ms.begin(),fifty_run_ms.end());
                median = (fifty_run_ms[24]+fifty_run_ms[25])/2;
                p95 = fifty_run_ms[47];
                result.write_benchmark_json(
                    output_path, model_load_ms, image_load_ms,
                    first_run_ms, median, p95
                );
                std::cout << "模型加载：" << model_load_ms << "ms\n"
                    << "图片处理：" << image_load_ms << "ms\n"
                    << "第一次运行：" << first_run_ms <<  "ms\n"
                    << "50次运行中位数：" << median << "ms\n"
                    << "p95 = " << p95 << "ms\n";
            }
        } else{
            std::vector<CsvRow> rows;
            rows.reserve(images.size());
            std::size_t success_count = 0;
            std::size_t failure_count = 0;
            std::size_t session_run_count = 0;
            double session_run_total_ms = 0;

            for (std::size_t i = 0; i < images.size(); ++i) {
                const auto& input_path = images[i];
                try {
                    const auto folder_name =
                        input_path.parent_path().filename().string() + input_path.stem().string();
                    const auto image_load_start = Clock::now();
                    const cv::Mat original_bgr = cv::imread(input_path.string(), cv::IMREAD_COLOR);
                    if (original_bgr.empty()) {
                        throw std::runtime_error("Cannot read or decode image: " + input_path.string());
                    }
                    const auto data = PreProcess::convert_bgr_to_chw(original_bgr, cv::Size(256, 256));
                    image_load_ms = std::chrono::duration<double, std::milli>(
                            Clock::now() - image_load_start).count();

                    auto prediction = engine.predict(data);
                    first_run_ms = prediction.run_ms;

                    ++session_run_count;
                    session_run_total_ms += prediction.run_ms;

                    const auto image_output_dir = output_path / folder_name;
                    if (!fs::create_directory(image_output_dir)) {
                        throw std::runtime_error("Output directory already exists: " +
                                                 image_output_dir.string());
                    }
                    PreProcess::save_input_tensor(data, image_output_dir, true);
                    result.write_images(image_output_dir, prediction, original_bgr);
                    result.write_json(image_output_dir, prediction);

                    rows.push_back({input_path, true, prediction.score,
                                    prediction.label, prediction.run_ms, ""});
                    ++success_count;
                    std::cout << "第" << i + 1 << "张图片的预测\n"
                              << "\t\tpred_score: " << prediction.score
                              << "\n\t\tpred_label: " << std::boolalpha << prediction.label
                              << "\n\t\tSession::Run: " << prediction.run_ms << " ms"
                              << "\n\t\tSaved: " << image_output_dir << '\n';
                    if(images.size() == 1 && args.benchmark){
                        //预热
                        int count = 5;
                        while(count--) prediction = engine.predict(data);
                        count = 50;
                        while(count--){
                            prediction = engine.predict(data);
                            fifty_run_ms[49-count] = prediction.run_ms;
                        }
                        sort(fifty_run_ms.begin(),fifty_run_ms.end());
                        median = (fifty_run_ms[24]+fifty_run_ms[25])/2;
                        p95 = fifty_run_ms[47];
                        result.write_benchmark_json(
                            image_output_dir, model_load_ms, image_load_ms,
                            first_run_ms, median, p95
                        );
                        std::cout << "模型加载：" << model_load_ms << "ms\n"
                            << "图片处理：" << image_load_ms << "ms\n"
                            << "第一次运行：" << first_run_ms <<  "ms\n"
                            << "50次运行中位数：" << median << "ms\n"
                            << "p95 = " << p95 << "ms\n";
                    }
                } catch (const std::exception& error) {
                    if (!is_batch) throw; // 单图模式直接失败；批量模式记录后继续。
                    rows.push_back({input_path, false, 0, false, 0, error.what()});
                    ++failure_count;
                    std::cerr << "Failed: " << input_path << ": " << error.what() << '\n';
                }
            }
            write_predictions_csv(output_path / "predictions.csv", rows);

            if (is_batch) {
                RunInfo info;
                info.model_path = model_path;
                info.image_list_path = fs::absolute(args.image_list);
                info.model_version = engine.model_version();
                info.image_count = images.size();
                info.success_count = success_count;
                info.failure_count = failure_count;
                info.session_run_count = session_run_count;
                info.model_load_ms = model_load_ms;
                info.session_run_total_ms = session_run_total_ms;
                info.batch_wall_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - batch_start).count();
                result.write_run_info_json(output_path, info);
                return failure_count == 0 ? 0 : 1;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
