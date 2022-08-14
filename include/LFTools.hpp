#ifndef LFTOOLS_HPP
#define LFTOOLS_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace cv;
using namespace std;

class lft{
    public:
        lft();
        ~lft();

        static void satFilt(Mat& Input, Mat& Output, int lowSat = 80, int highSat = 255);
        static void hsvFilt(Mat& Input, Mat& Output, Scalar low = Scalar(100,140,100), Scalar high = Scalar(110, 150, 190));
        static void sobelGrad(Mat& Input, Mat& Output);
        static void laplaceGrad(Mat& Input, Mat& Output);
        static void cannyGrad(Mat& Input, Mat& Output);

        static vector<Point> findMainContour(Mat& Input);
        static Point2f findCenter(Mat& Input);
        static double calcYaw(Point2f origin, Point2f center);
        static double calcContourYaw(Mat& Input);

        static void drawHoughLines(Mat& Input, double rho=1, double theta=CV_PI/180, double thresh=150, double srn=0, double stn=0);
        static void drawPHoughLines(Mat& Input, double rho=1, double theta=CV_PI/180, double thresh=50, double minL=50, double maxG=10);
        static void hsvTuner(Mat& Input);
    private:
        static int maxContourIndex(vector<vector<Point>> contours);
        static string mat_type2encoding(int mat_type){
            switch (mat_type) {
                case CV_8UC1:
                    return "mono8";
                case CV_8UC3:
                    return "bgr8";
                case CV_16SC1:
                    return "mono16";
                case CV_8UC4:
                    return "rgba8";
                default:
                    throw std::runtime_error("Unsupported encoding type");
            }
        }

};

#endif