#include "inspectflow/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/version.hpp>

namespace fs = std::filesystem;

namespace {

void write_bytes(const fs::path& path, const void* data, std::size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    stream.close();
}

}  // namespace

/*read_image用于在image输入模式中读取图片并进行预处理
包含读图-转RGB-resize大小-归一化-写到vector<float>中（由HWC转为CHW
*/
std::vector<float> PreProcess::read_image(const fs::path& path) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);

    if (image.empty()) {
        throw std::runtime_error(
            "Cannot read or decode image: " + path.string()
        );
    }
    return convert_bgr_to_chw(image, cv::Size(256, 256));
}

std::vector<float> PreProcess::convert_bgr_to_chw(const cv::Mat& image,
                                                   cv::Size target_size) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    if (target_size.width <= 0 || target_size.height <= 0) {
        throw std::invalid_argument("Target image size must be positive");
    }
    cv::Mat rgb;
    cv::cvtColor(image,rgb,cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, target_size, 0, 0, cv::INTER_LINEAR);
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
std::vector<float> PreProcess::read_input_bin(const fs::path& path) {
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

void PreProcess::save_input_tensor(
    const std::vector<float>& data,
    const fs::path& output_dir,
    bool from_image
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

    constexpr std::size_t expected_elements = 3 * 256 * 256;
    if (data.size() != expected_elements) {
        throw std::invalid_argument("Input must contain 3*256*256 float values");
    }
    for (const float value : data) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            throw std::invalid_argument("Input must contain finite values in [0, 1]");
        }
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
        << "  \"source_mode\": \"" << (from_image ? "image" : "input_bin") << "\"";

    if (from_image) {
        metadata
            << ",\n  \"read_mode\": \"imread IMREAD_COLOR -> BGR2RGB\""
            << ",\n  \"resize\": \"OpenCV INTER_LINEAR\""
            << ",\n  \"library\": \"opencv\""
            << ",\n  \"library_version\": \"" << CV_VERSION << "\""
            << ",\n  \"normalization\": "
            << "\"convertTo CV_32F, scale=1/255; "
            << "no external ImageNet normalization\"";
    }
    metadata << "\n}\n";

    const std::string text = metadata.str();

    write_bytes(
        metadata_path,
        text.data(),
        text.size()
    );
}
