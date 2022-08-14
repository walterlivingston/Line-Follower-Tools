#include "LFTools.hpp"

void lft::satFilt(Mat& Input, Mat& Output, int lowSat, int highSat){
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

void lft::hsvFilt(Mat& Input, Mat& Output, Scalar low, Scalar high){
    Mat hsv;
    cvtColor(Input, hsv, COLOR_BGR2HSV);
    inRange(Input, low, high, Output);
}

void lft::sobelGrad(Mat& Input, Mat& Output){
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

void lft::laplaceGrad(Mat& Input, Mat& Output){
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

void lft::cannyGrad(Mat& Input, Mat& Output){
    Mat Input_Gray;
    cvtColor(Input, Input_Gray, COLOR_BGR2GRAY);
    // Applying Gaussian Blur to deal with smaller contours
    GaussianBlur(Input_Gray, Input_Gray, Size(5,5), 0); //$$ Kernal Size (Default is 3,3) Increasing Makes Blurier. Note: Must be odd number $$
    Canny(Input_Gray, Output, 50, 200, 3);
}

vector<Point> lft::findMainContour(Mat& Input){    
    Mat sat;
    satFilt(Input, sat);
    Mat grad;
    hsvFilt(Input, grad);
    Mat comb;
    bitwise_or(sat, grad, comb);

    vector<vector<Point>> contours;
    findContours(comb, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    int max_index = maxContourIndex(contours);
    drawContours(Input, contours, max_index, Scalar(0,255,0), 3);
    return contours.at(max_index);
}

// (TODO) ADD RECTANGLE OF INTEREST
Point2f lft::findCenter(Mat& Input){
    vector<Point> contour = findMainContour(Input);
    RotatedRect rect = minAreaRect(contour);
    Point2f corners[4];
    rect.points(corners);

    double cx = rect.center.x;
    double cy = rect.boundingRect().y + rect.boundingRect().height/2;

    line(Input, Point(cx, 0), Point(cx, Input.size().height), Scalar(255,0,0), 3);
    line(Input, Point(0, cy), Point(Input.size().width, cy), Scalar(255,0,0), 3);

    for(int i = 0; i < 4; i++)
        line(Input, corners[i], corners[(i+1)%4], Scalar(0,0,255), 3);

    return Point2f(cx, cy);
}

double lft::calcYaw(Point2f origin, Point2f center){
    return atan((center.x - origin.x) / (center.y - origin.y));
}

double lft::calcContourYaw(Mat& Input){
    vector<Point> contour = findMainContour(Input);
    RotatedRect rect = minAreaRect(contour);
    Point2f corners[4];
    rect.points(corners);

    Point2f bl = corners[0];
    Point2f tl = corners[1];
    Point2f tr = corners[2];
    Point2f br = corners[3];
    double bcx = (bl.x + br.x) / 2;
    double bcy = (bl.y + br.y) / 2;
    double tcx = (tl.x + tr.x) / 2;
    double tcy = (tl.y + tr.y) / 2;

    double dist1 = sqrt(pow(bcx - tcx, 2) + pow(bcy - tcy, 2));

    double lmx = (tl.x + bl.x) / 2; // middle of left side x
    double lmy = (tl.y + bl.y) / 2; // middle of left side y
    double rmx = (tr.x + br.x) / 2; // middle of right side x
    double rmy = (tr.y + br.y) / 2; // middle of right side y

    double dist2 = sqrt(pow(lmx - rmx, 2) + pow(lmy - rmy, 2));

    vector<Point2f> clp;
    if(abs(dist1) > abs(dist2)){
        clp.push_back(Point2f(bcx, bcy));
        clp.push_back(Point2f(tcx, tcy));
    }else{
        clp.push_back(Point2f(rmx, rmy));
        clp.push_back(Point2f(lmx, lmy));
    }

    double contourYaw = calcYaw(clp.at(0), clp.at(1));

    for(int i = 0; i < 4; i++){
        line(Input, corners[i], corners[(i+1)%4], Scalar(0,0,255), 3);
    }
    line(Input, clp.at(0), clp.at(1), Scalar(0,0,255), 3);
    
    return contourYaw;
}

void lft::drawHoughLines(Mat& Input, double rho, double theta, double thresh, double srn, double stn){
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

void lft::drawPHoughLines(Mat& Input, double rho, double theta, double thresh, double minL, double maxG){
    Mat can;
    cannyGrad(Input, can);

    vector<Vec4i> lines; // will hold the results of the detection
    HoughLinesP(can, lines, rho, theta, thresh, minL, maxG); // runs the actual detection

    for(size_t i = 0; i < lines.size(); i++){
        Vec4i l = lines[i];
        line(Input, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
    }
}

void lft::hsvTuner(Mat& Input){
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

int lft::maxContourIndex(vector<vector<Point>> contours){
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