#include "inspectflow/preprocess.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace {

namespace fs = std::filesystem;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        std::random_device random;
        const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch().count();
        for (int attempt = 0; attempt < 10; ++attempt) {
            path_ = fs::temp_directory_path() /
                    ("inspectflow-preprocess-test-" + std::to_string(timestamp) +
                     "-" + std::to_string(random()));
            if (fs::create_directory(path_)) {
                return;
            }
        }
        throw std::runtime_error("Cannot create temporary test directory");
    }

    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

void write_file(const fs::path& path, const void* data, std::size_t bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create test file: " + path.string());
    }
    output.write(static_cast<const char*>(data),
                 static_cast<std::streamsize>(bytes));
    if (!output) {
        throw std::runtime_error("Cannot write test file: " + path.string());
    }
}

template <typename Operation>
void expect_error_containing(Operation operation, const std::string& expected) {
    try {
        operation();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(expected) != std::string::npos) {
            return;
        }
        throw std::runtime_error("Expected error containing '" + expected +
                                 "', got: " + error.what());
    }
    throw std::runtime_error("Expected error containing '" + expected +
                             "', but no exception was thrown");
}

void expect_near(float actual, float expected, const std::string& location) {
    if (std::abs(actual - expected) > 1e-6f) {
        throw std::runtime_error(location + ": unexpected channel value");
    }
}

void test_bgr_to_rgb_and_hwc_to_chw() {
    cv::Mat image(2, 2, CV_8UC3);
    image.at<cv::Vec3b>(0, 0) = {10, 20, 30};
    image.at<cv::Vec3b>(0, 1) = {40, 50, 60};
    image.at<cv::Vec3b>(1, 0) = {70, 80, 90};
    image.at<cv::Vec3b>(1, 1) = {100, 110, 120};

    const auto actual = PreProcess::convert_bgr_to_chw(image, {2, 2});
    if (actual.size() != 12) {
        throw std::runtime_error("Expected three channels with four pixels each");
    }

    const std::array<float, 12> expected{
        30, 60, 90, 120,   // R: row-major
        20, 50, 80, 110,   // G: row-major
        10, 40, 70, 100,   // B: row-major
    };
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expect_near(actual[i], expected[i] / 255.0f,
                    "CHW index " + std::to_string(i));
    }
}

void test_read_input_bin(const fs::path& directory) {
    const auto short_path = directory / "short.f32";
    const std::array<char, 4> short_data{{0, 0, 0, 0}};
    write_file(short_path, short_data.data(), short_data.size());
    expect_error_containing(
        [&] { PreProcess::read_input_bin(short_path); }, "786432 bytes");

    constexpr std::size_t element_count = 3 * 256 * 256;
    std::vector<float> data(element_count, 0.5f);
    data[0] = std::numeric_limits<float>::quiet_NaN();
    const auto nan_path = directory / "nan.f32";
    write_file(nan_path, data.data(), data.size() * sizeof(float));
    expect_error_containing(
        [&] { PreProcess::read_input_bin(nan_path); }, "NaN or Inf");

    data[0] = 0.5f;
    const auto valid_path = directory / "valid.f32";
    write_file(valid_path, data.data(), data.size() * sizeof(float));
    const auto actual = PreProcess::read_input_bin(valid_path);
    if (actual.size() != element_count) {
        throw std::runtime_error("Valid binary input has wrong element count");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != 0.5f) {
            throw std::runtime_error("Valid binary input has wrong value at index " +
                                     std::to_string(i));
        }
    }
}

void test_unreadable_images(const fs::path& directory) {
    expect_error_containing(
        [&] { PreProcess::read_image(directory / "missing.png"); },
        "Cannot read or decode image");

    const auto corrupt_path = directory / "corrupt.png";
    const std::string invalid_png = "This is not a PNG image";
    write_file(corrupt_path, invalid_png.data(), invalid_png.size());
    expect_error_containing(
        [&] { PreProcess::read_image(corrupt_path); },
        "Cannot read or decode image");
}

}  // namespace

int main() {
    try {
        const TemporaryDirectory temporary_directory;
        test_bgr_to_rgb_and_hwc_to_chw();
        test_read_input_bin(temporary_directory.path());
        test_unreadable_images(temporary_directory.path());
        std::cout << "Preprocessing test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Preprocessing test failed: " << error.what() << '\n';
        return 1;
    }
}
