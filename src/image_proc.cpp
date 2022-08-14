#include "image_proc.hpp"

void imp::satFilt(Mat& Input, Mat& Output, int lowSat = 80, int highSat = 255){
    // Intialization.
    cv::Mat hls;
    // Convert Color to HLS Colorspace
    cv::cvtColor(Input, hls, CV_BGR2HLS);
    // Initializing 3 individual single channel images.
    Mat hlsSplit[3];
    // Splitting
    cv::split(hls, hlsSplit);
    // Getting the Saturation Channel
    Mat chan = hlsSplit[2];
    // Thresholding to remove lower saturations.
    threshold(chan, Output, lowSat, highSat, THRESH_BINARY);// $$ Min and Max Threshold (80 and 255 by default) $$
}

void imp::hsvFilt(Mat& Input, Mat& Output, Scalar low = Scalar(100,140,100), Scalar high = Scalar(110, 150, 190)){
    Mat hsv;
    cvtColor(Input, hsv, COLOR_BGR2HSV);
    inRange(Input, low, high, Output);
}

void imp::sobelGrad(Mat& Input, Mat& Output){
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad;

    cv::Mat Input_Gray;
    cvtColor(Input, Input_Gray, COLOR_BGR2GRAY);
    // Applying Gaussian Blur to deal with smaller contours:
    GaussianBlur(Input_Gray, Input_Gray, Size(3,3), 0, 0); // $$ Kernal Size (Default is 3,3) Increasing Makes Blurier. Note: Must be odd number $$
    // Computing Sobel:
    // Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT); [opencv Function Definitions]
        Sobel(Input_Gray, grad_x, CV_16S, 1, 0, 3, 1, 0);  // $$ Ksize (3 By Default) $$
        Sobel(Input_Gray, grad_y, CV_16S, 0, 1, 3, 1, 0);
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    // Thresholding Edges (Weaker Edges Removed):
    threshold(grad, Output, 70, 255, THRESH_BINARY); // $$ Min and Max Threshold (90 and 255 by default) $$
    // Dialating to Fill in a bit: 
    dilate(grad, grad, Mat(), Point(-1, -1), 2, 1, 1); // $$ This defaults to 3x3 Kernal for the convolution $$
}

void imp::laplaceGrad(Mat& Input, Mat& Output){
    cv::Mat Input_Gray;
    cvtColor(Input, Input_Gray, COLOR_BGR2GRAY);
    // Applying Gaussian Blur to deal with smaller contours
    GaussianBlur(Input_Gray, Input_Gray, Size(5,5), 0); //$$ Kernal Size (Default is 3,3) Increasing Makes Blurier. Note: Must be odd number $$
    Mat lap;
    // Laplacian(src_gray, laplacian, ddepth, ksize); [opencv Function Definitions]
    Laplacian(Input_Gray, lap, CV_64F, 3);
    // Convert back to CV_8U for finding contours
    lap.convertTo(Output, CV_8U);    
}

void imp::cannyGrad(Mat& Input, Mat& Output){
    Mat Input_Gray;
    cvtColor(Input, Input_Gray, COLOR_BGR2GRAY);
    // Applying Gaussian Blur to deal with smaller contours
    GaussianBlur(Input_Gray, Input_Gray, Size(5,5), 0); //$$ Kernal Size (Default is 3,3) Increasing Makes Blurier. Note: Must be odd number $$
    Canny(Input_Gray, Output, 50, 200, 3);
}

void imp::drawMainContour(Mat& Input){    
    Mat sat;
    satFilt(Input, sat);
    Mat grad;
    hsvFilt(Input, grad);
    Mat comb;
    bitwise_or(sat, grad, comb);

    vector<vector<Point>> contours;
    findContours(comb, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    drawContours(Input, contours, maxContourIndex(contours), Scalar(0,255,0), 3);
}

void imp::drawHoughLines(Mat& Input, double rho, double theta, double thresh, double srn, double stn){
    Mat can;
    cannyGrad(Input, can);

    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(can, lines, rho, theta, thresh, srn, stn); // runs the actual detection

    for(size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line(Input, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
    }
}

void imp::drawPHoughLines(Mat& Input, double rho, double theta, double thresh, double minL, double maxG){
    Mat can;
    cannyGrad(Input, can);

    vector<Vec4i> lines; // will hold the results of the detection
    HoughLinesP(can, lines, rho, theta, thresh, minL, maxG); // runs the actual detection

    for(size_t i = 0; i < lines.size(); i++){
        Vec4i l = lines[i];
        line(Input, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
    }
}

void imp::hsvTuner(Mat& Input){
    //import image
    cv::Mat original = Input.clone();
    //Convert Image Color
    cvtColor(Input, Input, CV_BGR2HSV);
    cv::imshow("HSV", Input);

    //Initializing Slider
    int Hmin=0, Hmax=256, Smin=0, Smax=256,Vmin=0,Vmax=256;
    cv::namedWindow("Sliders",cv::WINDOW_NORMAL);
    cv::resizeWindow("Sliders",300,300);
    cv::createTrackbar("Hue Min", "Sliders", &Hmin, 256) ;
    cv::createTrackbar("Hue Max", "Sliders", &Hmax, 256) ;
    cv::createTrackbar("Saturation Min","Sliders", &Smin,256);
    cv::createTrackbar("Saturation Max","Sliders", &Smax,256);
    cv::createTrackbar("Value Min","Sliders", &Vmin,256);
    cv::createTrackbar("Value Max","Sliders", &Vmax,256);

    while (true){
        cv::Mat Filtered = Input.clone();
        //Filtering Hue, Saturation and Value
        cv::inRange(Filtered, cv::Scalar(Hmin,Smin,Vmin), cv::Scalar(Hmax,Smax,Vmax),Filtered);
        cv::namedWindow("Filtered",cv::WINDOW_AUTOSIZE);
        cv::imshow("Filtered",Filtered);
        cv::waitKey(5);
    }
}

int imp::maxContourIndex(vector<vector<Point>> contours){
    double c_max = 0;
    int max_index = -1;
    for (int i = 0; i < (int)contours.size(); i++) {
        if (contourArea(contours.at(i)) > c_max) {
            c_max = contourArea(contours.at(i));
            max_index = i;
        }
    }
    return max_index;
}