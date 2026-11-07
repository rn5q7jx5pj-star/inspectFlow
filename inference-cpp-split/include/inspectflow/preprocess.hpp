/*用于预处理输入图片：处理image或者二进制文件
read_image：输入图片，输出数据
read_input_bin：输入二进制文件，输出数据
save_input_tensor：保存中间值 用于查找错误
*/
#pragma once

#include <filesystem>
#include <vector>
#include <opencv2/core.hpp>

class PreProcess {
public:
    // 读取图片或已经预处理好的 float32 二进制输入。
    static std::vector<float> convert_bgr_to_chw(const cv::Mat& image,
                                                  cv::Size target_size);
    static std::vector<float> read_image(const std::filesystem::path& path);
    static std::vector<float> read_input_bin(const std::filesystem::path& path);

    // 保存实际传给模型的张量；from_image 用于记录正确的输入来源。
    static void save_input_tensor(const std::vector<float>& data,
                                  const std::filesystem::path& output_dir,
                                  bool from_image);
};
