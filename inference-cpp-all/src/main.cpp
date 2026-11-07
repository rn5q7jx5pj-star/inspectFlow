#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "binary_inference.hpp"

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

/*print_tensor_info接收模型内部变量的名字的指针，以及对应的tensor对象
功能是打印输入/输出的内部名称+dtype+shape
*/
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
        << "  [--image <image.png> | --input-bin <input.f32>]\n"
        << "  [--output-dir <new-directory>]\n"
        << "Inference requires --output-dir.\n";
}

/*主函数：用于模型模型内容的解析 然后传入模型推理函数
*/
int main(int argc, char* argv[]) {
    if (argc == 2 && (std::string(argv[1]) == "--help" ||
                      std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        return 0;
    }
    try {
        const auto args = parse_inference_args(argc, argv);
        const auto model_path = std::filesystem::absolute(args.model);
        if (!std::filesystem::is_regular_file(model_path)) {
            std::cerr << "Model file does not exist or is not a regular file: "
                      << model_path << '\n';
            return 1;
        }

        std::cout << "ONNX Runtime version: "
                  << OrtGetApiBase()->GetVersionString()
                  << '\n';

        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "InspectFlow"
        );
        Ort::SessionOptions options;
        Ort::Session session(env, model_path.c_str(), options);
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

        if (!args.image.empty() || !args.input_bin.empty()) {
            run_binary_inference(session, allocator, args);
        }
        return 0;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 2;
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: "
                  << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
