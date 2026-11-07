#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>
#include <locale>
#include <limits>
#include <sstream>
#include <atomic>
#include <cstdint>

#include "httplib.h"
#include "inspectflow/inference_engine.hpp"
#include "inspectflow/preprocess.hpp"
#include "inspectflow/detection_record.hpp"
#include "inspectflow/detection_store.hpp"
#include <opencv2/imgcodecs.hpp>

void write_json_string(std::ostream& out, const std::string& value) {
    constexpr char hex[] = "0123456789abcdef";
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            out << '\\' << static_cast<char>(c);
        } else if (c < 0x20) {
            out << "\\u00" << hex[c >> 4] << hex[c & 0x0f];
        } else {
            out << static_cast<char>(c);
        }
    }
    out << '"';
}

std::string next_request_id() {
    static std::atomic<std::uint64_t> counter{0};

    const auto number =
        counter.fetch_add(1, std::memory_order_relaxed) + 1;

    return "req-" + std::to_string(number);
}

void send_error(
    httplib::Response& response,
    int status,
    const std::string& request_id,
    const char* code,
    const char* message
) {
    // 本示例的ID由程序生成，code、message使用下面列出的固定文本。
    const std::string body =
        "{\"request_id\":\"" + request_id +
        "\",\"error\":{\"code\":\"" + code +
        "\",\"message\":\"" + message + "\"}}";

    response.status = status;
    response.set_content(body, "application/json");
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: inspectflow_server <model.onnx> <port>\n";
        return 1;
    }

    try {
        const std::filesystem::path model_path = argv[1];

        // 将端口字符串转换成整数，并检查是否合法。
        const std::string port_text = argv[2];
        std::size_t used = 0;
        const int port = std::stoi(port_text, &used);

        DetectionStore store(argv[3]);

        if (used != port_text.size() || port < 1 || port > 65535) {
            throw std::invalid_argument("Invalid port");
        }

        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error("Model file does not exist");
        }

        // 这里会加载模型。失败则抛异常，后面的监听不会执行。
        InferenceEngine engine(model_path);

        httplib::Server server;

        // 注册路由：收到 GET /health 时，执行这个函数。
        server.Get(
            "/health",
            [](const httplib::Request&,
               httplib::Response& response) {
                const std::string request_id = next_request_id();

                // 同时放进响应头，方便调试。
                response.set_header("X-Request-ID", request_id);
                response.status = 200;
                response.set_content(
                    R"({"status":"ok","model_loaded":true})",
                    "application/json"
                );
            }
        );

        //注册路由：
        // 整个HTTP请求体最多6MiB，给multipart格式本身留一些空间。
        server.set_payload_max_length(6 * 1024 * 1024);

        server.Post(
            "/v1/detections",
            [&engine, &store](const httplib::Request& request,
            httplib::Response& response) {
                const std::string request_id = next_request_id();

                // 同时放进响应头，方便调试。
                response.set_header("X-Request-ID", request_id);

                // 1. 检查是否为multipart表单，以及是否包含image文件。
                if (!request.is_multipart_form_data() ||
                    !request.form.has_file("image")) {

                    send_error(
                        response,
                        400,
                        request_id,
                        "MISSING_IMAGE",
                        "请求中缺少image文件"
                    );
                    return;
                }

                // 2. 取出上传文件。
                const auto file = request.form.get_file("image");

                // content保存图片文件的原始字节。
                constexpr std::size_t max_image_bytes = 5 * 1024 * 1024;

                //检查：上传的文件内容是否为空，也就是0字节文件
                if (file.content.empty()) {
                    send_error(
                        response,
                        400,
                        request_id,
                        "IMAGE_IS_EMPTY",
                        "图片文件内容为空"
                    );
                    return;
                }

                if (file.content.size() > max_image_bytes) {
                    send_error(
                        response,
                        413,
                        request_id,
                        "IMAGE_TOO_LARGE",
                        "图片大小超过5MiB"
                    );
                    return;
                }

                // 3. 将文件字节放进OpenCV可以读取的缓冲区。
                const std::vector<unsigned char> bytes(
                    file.content.begin(),
                    file.content.end()
                );

                // 4. 解码PNG/JPEG等文件，得到BGR像素矩阵。
                cv::Mat image;

                try {
                    image = cv::imdecode(bytes, cv::IMREAD_COLOR);
                } catch (const cv::Exception&) {//检查：OpenCV解码时是否抛出了异常。
                    send_error(
                        response,
                        400,
                        request_id,
                        "INVALID_IMAGE",
                        "无法解码上传的图片"
                    );
                    return;
                }

                if (image.empty()) {
                    send_error(
                        response,
                        400,
                        request_id,
                        "INVALID_IMAGE",
                        "无法解码上传的图片"
                    );
                    return;
                }

                try {
                    auto input = PreProcess::convert_bgr_to_chw(
                        image,
                        cv::Size(256, 256)
                    );

                    const auto prediction = engine.predict(input);

                    DetectionRecord record;
                    record.request_id = request_id;
                    record.model_name = engine.get_ModelName();
                    record.model_version = engine.model_version();
                    record.pred_score = prediction.score;
                    record.pred_label = prediction.label;
                    record.inference_ms = prediction.run_ms;

                    std::int64_t id;
                    try {
                        id = store.insert(record);
                    } catch (const std::exception& error) {
                        std::cerr << "[" << request_id << "] "
                                  << "Save failed: " << error.what() << '\n';
                        send_error(
                            response,
                            500,
                            request_id,
                            "RESULT_SAVE_FAILED",
                            "检测结果保存失败"
                        );
                        return;
                    }

                    std::ostringstream json;
                    json.imbue(std::locale::classic());

                    json << std::setprecision(9)
                        << std::boolalpha
                        << "{"
                        << "\"request_id\":\"" << request_id << "\","
                        << "\"detection_id\":\"" << id << "\","
                        << "\"pred_score\":" << prediction.score << ","
                        << "\"pred_label\":" << prediction.label << ","
                        << "\"inference_ms\":" << prediction.run_ms
                        << "}";

                    response.status = 200;
                    response.set_content(json.str(), "application/json");

                } catch (const std::exception& error) {
                    // 详细原因记在服务器终端，带上同一个请求ID。
                    std::cerr << "[" << request_id << "] "
                            << "Detection failed: "
                            << error.what() << '\n';

                    // 客户端收到稳定、简短的错误。
                    send_error(
                        response,
                        500,
                        request_id,
                        "INFERENCE_FAILED",
                        "检测处理失败"
                    );

                } catch (...) {
                    std::cerr << "[" << request_id << "] "
                            << "Unknown detection error\n";

                    send_error(
                        response,
                        500,
                        request_id,
                        "INTERNAL_ERROR",
                        "服务器内部错误"
                    );
                }
            }
        );

        server.Get("/v1/models/current",
            [&engine](const httplib::Request&, httplib::Response& response) {
                const std::string request_id = next_request_id();

                // 同时放进响应头，方便调试。
                response.set_header("X-Request-ID", request_id);
                std::ostringstream json;

                json << "{\"model_name\":";
                write_json_string(json, engine.get_ModelName());

                // 使用 ONNX 元数据版本；非正值与 ORT 占位值均不作有效版本。
                const auto model_version = engine.model_version();
                json << ",\"model_version\":";
                if (model_version > 0 &&
                    model_version != std::numeric_limits<std::int64_t>::max()) {
                    json << model_version;
                } else {
                    json << "null";
                }
                json << ",\"input_name\":";
                write_json_string(json, engine.get_InputName());

                json << ",\"input_shape\":[";
                const auto& shape = engine.get_shape();
                for (std::size_t i = 0; i < shape.size(); ++i) {
                    if (i != 0) json << ',';
                    json << shape[i];
                }

                json << "],\"input_dtype\":";
                write_json_string(json,
                    engine.get_dtype() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                        ? "float32" : "unknown");

                json << ",\"provider\":\"CPUExecutionProvider\""
                    << ",\"onnxruntime_version\":";
                write_json_string(json, Ort::GetVersionString());
                json << '}';

                response.status = 200;
                response.set_content(json.str(), "application/json");
            }
        );

        std::cout << "Model loaded. Starting server on "
                  << "127.0.0.1:" << port << std::endl;

        // 开始监听。正常运行期间，程序会停留在这里处理请求。
        if (!server.listen("127.0.0.1", port)) {
            throw std::runtime_error(
                "Cannot start server; check whether the port is occupied"
            );
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Startup failed: " << error.what() << '\n';
        return 1;
    }
}
