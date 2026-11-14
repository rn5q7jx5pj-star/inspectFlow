/*用于加载模型并进行多次预测
InferenceEngine为构造函数 在创建Engine对象时就会加载模型
predict用于推理函数 输入数据输出一个自定义结构Prediction用于存储结果
*/
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>
#include <string>

#include <onnxruntime_cxx_api.h>

struct Prediction {
    float score;
    bool label;
    std::vector<float> anomaly_map;
    std::vector<std::uint8_t> mask;
    double run_ms;
};

class InferenceEngine {
public:
    //加载模型｜一次加载
    explicit InferenceEngine(
        const std::filesystem::path& model_path,
        int intra_op_threads = 0
    );
    //模型推理｜多次推理
    Prediction predict(const std::vector<float>& data);
    std::int64_t model_version() const;
    const std::string& get_ModelName() const { return model_name_; }
    const std::string& get_InputName() const { return input_name_; }
    const std::vector<std::int64_t>& get_shape() const { return shape_; }
    ONNXTensorElementDataType get_dtype() const { return dtype_; }

private:
    Ort::Env env_;
    Ort::Session session_;

    std::string model_name_;
    std::string input_name_;
    std::vector<std::int64_t> shape_;
    ONNXTensorElementDataType dtype_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
};
