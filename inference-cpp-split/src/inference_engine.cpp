#include <cstdint>
#include <onnxruntime_cxx_api.h>

#include "inspectflow/inference_engine.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>


//namespace { ... } 叫匿名命名空间。在里面放了两类只供这个 .cpp 使用的东西
namespace {

constexpr std::size_t kPixels = 256 * 256;
constexpr std::size_t kInputElements = 3 * kPixels;
const std::array<int64_t, 4> kInputShape{1, 3, 256, 256};
const std::array<const char*, 4> kOutputNames{
    "pred_score", "pred_label", "anomaly_map", "pred_mask"};

void require_tensor_info(const Ort::TypeInfo& type_info,
                         ONNXTensorElementDataType element_type,
                         const std::vector<int64_t>& shape,
                         const std::string& name) {
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        throw std::runtime_error(name + " is not a tensor");
    }
    const auto info = type_info.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != element_type || info.GetShape() != shape) {
        throw std::runtime_error(name + " has an unexpected dtype or shape");
    }
}

void require_tensor_value(const Ort::Value& value,
                          ONNXTensorElementDataType element_type,
                          const std::vector<int64_t>& shape,
                          const char* name) {
    if (!value.IsTensor()) {
        throw std::runtime_error(std::string(name) + " is not a tensor");
    }
    const auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != element_type || info.GetShape() != shape) {
        throw std::runtime_error(std::string(name) + " has an unexpected dtype or shape");
    }
}

Ort::SessionOptions make_session_options(int intra_op_threads) {
    if (intra_op_threads < 0) {
        throw std::invalid_argument("intra_op_threads cannot be negative");
    }

    Ort::SessionOptions options;

    if (intra_op_threads > 0) {
        options.SetIntraOpNumThreads(intra_op_threads);
    }

    return options;
}

}  // namespace

// 模型只加载一次；提前确认它符合当前固定的输入、输出协议。Ort::SessionOptions为配置
InferenceEngine::InferenceEngine(
    const std::filesystem::path& model_path,
    int intra_op_threads
)
    : env_(ORT_LOGGING_LEVEL_WARNING, "InspectFlow"),
      session_(
          env_,
          model_path.c_str(),
          make_session_options(intra_op_threads)
      )  {
    if (session_.GetInputCount() != 1) {
        throw std::runtime_error("Expected exactly one model input");
    }
    require_tensor_info(session_.GetInputTypeInfo(0),
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                        {1, 3, 256, 256}, "Model input");

    
    model_name_ = model_path.filename().string();

    Ort::AllocatorWithDefaultOptions allocator;
    auto name = session_.GetInputNameAllocated(0, allocator);
    input_name_ = name.get();

    auto type_info = session_.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    shape_ = tensor_info.GetShape();
    dtype_ = tensor_info.GetElementType();
    
    for (const auto* required_name : kOutputNames) {
        bool found = false;
        for (std::size_t i = 0; i < session_.GetOutputCount(); ++i) {
            const auto name = session_.GetOutputNameAllocated(i, allocator);
            if (std::string(name.get()) != required_name) continue;

            const bool is_map = std::string(required_name) == "anomaly_map";
            const bool is_mask = std::string(required_name) == "pred_mask";
            require_tensor_info(session_.GetOutputTypeInfo(i),
                                (is_map || std::string(required_name) == "pred_score")
                                    ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
                                (is_map || is_mask)
                                    ? std::vector<int64_t>{1, 1, 256, 256}
                                    : std::vector<int64_t>{1},
                                required_name);
            found = true;
            break;
        }
        if (!found) {
            throw std::runtime_error(std::string("Missing model output: ") + required_name);
        }
    }
}

std::int64_t InferenceEngine::model_version() const {
    return session_.GetModelMetadata().GetVersion();
}

// data 必须是预处理后的 float32、NCHW、[1,3,256,256]；不负责读图或保存结果。
Prediction InferenceEngine::predict(const std::vector<float>& data) {
    if (data.size() != kInputElements) {
        throw std::invalid_argument("Input must contain 3*256*256 float values");
    }
    for (const float value : data) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Input contains a non-finite value");
        }
    }

    // CreateTensor 借用可写内存；复制一份以保留 predict 的 const 输入接口。
    auto input_data = data;
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory, input_data.data(), input_data.size(),
        kInputShape.data(), kInputShape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session_.GetInputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};

    const auto start = std::chrono::steady_clock::now();
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                kOutputNames.data(), kOutputNames.size());
    const double run_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    require_tensor_value(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                         {1}, kOutputNames[0]);
    require_tensor_value(outputs[1], ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
                         {1}, kOutputNames[1]);
    require_tensor_value(outputs[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                         {1, 1, 256, 256}, kOutputNames[2]);
    require_tensor_value(outputs[3], ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
                         {1, 1, 256, 256}, kOutputNames[3]);

    Prediction prediction{};
    prediction.score = outputs[0].GetTensorData<float>()[0];
    prediction.label = outputs[1].GetTensorData<bool>()[0];
    prediction.run_ms = run_ms;
    if (!std::isfinite(prediction.score)) {
        throw std::runtime_error("Non-finite pred_score");
    }

    // 结果要脱离 outputs 的生命周期，因此不能直接保存 GetTensorData 返回的指针。
    const float* map = outputs[2].GetTensorData<float>();
    const bool* mask = outputs[3].GetTensorData<bool>();
    prediction.anomaly_map.reserve(kPixels);
    prediction.mask.reserve(kPixels);
    for (std::size_t i = 0; i < kPixels; ++i) {
        if (!std::isfinite(map[i])) {
            throw std::runtime_error("Non-finite anomaly_map");
        }
        prediction.anomaly_map.push_back(map[i]);
        prediction.mask.push_back(mask[i] ? 1 : 0);
    }
    return prediction;
}
