#include "image_proc.hpp"

int main(int argc, char** argv){
    VideoCapture vid = VideoCapture(0);
    Mat frame;
    while(true){
        vid.read(frame);
        imp::drawHoughLines(frame);
        imshow("Webcam", frame);
        waitKey(1);
    }
    vid.release();
    cv::destroyAllWindows();
    return 0;
}