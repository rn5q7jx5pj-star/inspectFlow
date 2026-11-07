/*用于预测后的后处理
write_json：输入路径和predict结果，保存预测报告（整体），输出json文件需要保存的字符串，调用write_bytes后保存为json文件
write_images：输入路径和predict结果，保存热力图和二值图
write_run_info_json：保存批量运行的模型、计数、版本和计时信息
write_bytes：用于写文件
*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

struct Prediction;

struct RunInfo {
    std::filesystem::path model_path;
    std::filesystem::path image_list_path;
    std::int64_t model_version = 0;
    std::size_t image_count = 0;
    std::size_t success_count = 0;
    std::size_t failure_count = 0;
    std::size_t session_run_count = 0;
    double model_load_ms = 0;
    double session_run_total_ms = 0;
    double batch_wall_ms = 0;
};

class Result {
public:
    // 将一次推理的数值摘要保存到已有目录中的 prediction.json。
    void write_json(const std::filesystem::path& output_dir,
                    const Prediction& prediction);

    // 将异常图和二值掩码分别保存为 heatmap.png、mask.png。
    void write_images(const std::filesystem::path& output_dir,
                      const Prediction& prediction);

    // 保存批量运行的模型、预处理、统计和计时信息。
    void write_run_info_json(const std::filesystem::path& output_dir,
                             const RunInfo& info);

    //benchmark
    void write_benchmark_json(const std::filesystem::path& output_dir,
                          double model_load_ms,
                          double preprocess_ms,
                          double first_run_ms,
                          double median_ms,
                          double p95_ms);

private:
    // 只负责写原始字节；不能直接用来生成 PNG 等图片格式。
    static void write_bytes(const std::filesystem::path& path,
                            const void* data, std::size_t bytes);
};
