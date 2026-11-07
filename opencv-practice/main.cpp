#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

int main(int argc, char* argv[]){
    if(argc != 2){
        std::cerr <<  "用法：opencv_check <图片路径>\n";
        return 1;
    }
    const cv::Mat image = cv::imread(argv[1],cv::IMREAD_COLOR);
    if(image.empty()){
        std::cerr << "图片读取失败：" << argv[1] << '\n';
        return 1;
    }
    std::cout << "openCV--version:" << CV_VERSION << "\n";
    std::cout << "width:" << image.cols << "\n";
    std::cout << "height:" << image.rows << "\n";
    std::cout << "roadNums:" << image.channels() << "\n";

    return 0;
}


