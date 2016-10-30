#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <typeinfo>
#include <ctime>
#include <cmath>
#include <thread>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace boost::asio::ip;

VideoCapture camera;
Mat frame, frame_grey;
std::string message  = "";
bool frameExists = false;
bool byColor = false;
char key;
int thresh = 100;
int id_thresh = 0;
int id_thresh_max = 0;
int max_thresh = 255;
int thresh_max = 100;
int lowH = 0, highH = 179;
int lowS = 0, highS = 255;
int lowV = 0, highV = 255;
int minContourSize = 25;
double image_t = 0, threshold_t = 0;
const double PI = 3.14159265358979323846264338328;

void thresh_callback(int, void*);
void denoiseMat(Mat, int);
double rectangularity(vector<Point>, Rect);
double circularity(vector<Point>, float);

int main(int argc, char *argv[]) {
  //printf("CUDA: %i", getCudaEnabledDeviceCount());
  VideoCapture camera;
  if (argc > 1 && boost::starts_with(argv[1], "http://")) {
    std::string address;
    address.append(argv[1]);
    address.append("video?x.mjpeg");
    camera = VideoCapture(address);
    byColor = true;
  } else {
    byColor = true; //boost::starts_with(argv[1], "color");
    camera = VideoCapture(CV_CAP_ANY);
  }

  boost::asio::io_service io_service;
  udp::resolver resolver(io_service);
  udp::resolver::query query(udp::v4(), argv[2], "3290");
  udp::endpoint receiver_endpoint = *resolver.resolve(query);
  udp::socket socket(io_service);
  socket.open(udp::v4());

  while (1) {
    image_t = (double)getTickCount();
    frameExists = camera.read(frame);

    if (!frameExists) {
      printf("Cannot read image from camera. \n");
      return -1;
    }

    key = waitKey(10);     // Capture Keyboard stroke
    if (char(key) == 27) {
      printf("\n");
      break;      // If you hit ESC key loop will break.
    }

    image_t = ((double)getTickCount() - image_t)/getTickFrequency();

    namedWindow("Image", CV_WINDOW_AUTOSIZE);
    imshow("Image", frame);


    createTrackbar("Canny Threshold:", "Image", &thresh, max_thresh, thresh_callback);
    createTrackbar("Canny Max", "Image", &thresh_max, max_thresh, thresh_callback);
    createTrackbar("LowH", "Image", &lowH, 179); //Hue (0 - 179)
    createTrackbar("HighH", "Image", &highH, 179);

    createTrackbar("LowS", "Image", &lowS, 255); //Saturation (0 - 255)
    createTrackbar("HighS", "Image", &highS, 255);

    createTrackbar("LowV", "Image", &lowV, 255); //Value (0 - 255)
    createTrackbar("HighV", "Image", &highV, 255);

    createTrackbar("Min Contour Size:", "Image", &minContourSize, 64);

    message = "";
    thresh_callback(0, 0);
    socket.send_to(boost::asio::buffer(message), receiver_endpoint);
  }

  waitKey(0);
  return 0;
}

/** @function thresh_callback */
void thresh_callback(int, void*) {
  Mat frame_hsv, frame_grey, frame_thresh, frame_thresh_circle, canny_output, drawing;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy, lines;
  vector<Vec3f> circles;

  if (byColor) {
    cvtColor(frame, frame_hsv, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    inRange(frame_hsv, Scalar(lowH, lowS, lowV), Scalar(highH, highS, highV), frame_thresh); //Threshold the image
    denoiseMat(frame_thresh, 3);
    GaussianBlur(frame_thresh, frame_thresh, Size(9, 9), 2, 2);
    HoughCircles(frame_thresh, circles, CV_HOUGH_GRADIENT,
      1, // accumulator resolution (size of image/2)
      frame_thresh.rows/8, // min distance between two circles
      thresh_max, // internal canny high threshold
      35, // threshold for centre detection
      0, 1000); // min max radius, 0 is default
      namedWindow("Thresholded Image", CV_WINDOW_AUTOSIZE);
      imshow("Thresholded Image", frame_thresh);

      // Detect edges using canny
      threshold_t = (double)getTickCount();
      Canny(frame_thresh, canny_output, thresh, thresh_max, 3);
      drawing = Mat::zeros(canny_output.size(), CV_8UC3);
  } else {
    cvtColor(frame, frame_grey, COLOR_BGR2GRAY); //Convert the captured frame from BGR to B&W
    GaussianBlur(frame_grey, frame_grey, Size(3, 3), 2, 2 );
    HoughCircles(frame_grey, circles, CV_HOUGH_GRADIENT,
      2, // accumulator resolution (size of image/2)
      frame_grey.rows/4, // min distance between two circles
      thresh_max, // internal canny high threshold
      50, // threshold for centre detection
      0, 200); // min max radius, 0 is default

      // Detect edges using canny
      Canny(frame_grey, canny_output, thresh, thresh_max, 3);
      namedWindow("Canny", CV_WINDOW_AUTOSIZE);
      imshow("Canny", canny_output);
      drawing = Mat::zeros(canny_output.size(), CV_8UC3);
  }


  // Find contours
  findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

  vector<vector<Point> > contours_poly(contours.size());
  vector<Point2f> centre(contours.size());
  vector<float> radius(contours.size());
  Scalar contour_colour(0, 0, 255);
  Scalar bound_colour(0, 255, 0);

  int largest_radius = -1;
  Point largest_circle;

  #pragma omp parallel for
  for (size_t i = 0; i < circles.size(); i++) {
    Point centre(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    if (radius > largest_radius) largest_circle = centre;
    circle(drawing, centre, 3, Scalar(0,0,255), -1, 4, 0 );
    circle(drawing, centre, radius, Scalar(0,0,255), -1, 4, 0);
  }
  message += "circle: ";
  message += std::to_string(largest_circle.x);
  message += ", ";
  message += std::to_string(largest_circle.y);
  message += "\n";

  double largest_area = -1;
  Point largest_rect;

  #pragma omp parallel for
  for (int i = 0; i < contours.size(); i++) {
    approxPolyDP(contours[i], contours_poly[i], 0.04 * arcLength(contours[i], true), true);

    vector<Point> temp_contour_circle = contours[i], temp_contour_poly_circle = contours_poly[i];
    minEnclosingCircle((Mat)contours_poly[i], centre[i], radius[i]);
    bool isCircle = (circularity(temp_contour_poly_circle, radius[i])) > 80;
    if (!isContourConvex(contours_poly[i]) || isCircle)
      continue;
    double area = contourArea(contours_poly[i]);
    if (area > largest_area) {
      Rect rect = boundingRect(contours_poly[i]);
      largest_rect = Point((rect.x + rect.width)/2, (rect.y + rect.height)/2);
    }
    if (contours_poly[i].size() == 4 && area >= minContourSize) {
      drawContours(drawing, contours_poly, i, bound_colour,
        -1, // line thickness
        8, //line type
        hierarchy,
        0, //max level to draw
        Point(0, 0)); // Point() offset
    }
  }
  message += "rect: ";
  message += std::to_string(largest_rect.x);
  message += ", ";
  message += std::to_string(largest_rect.y);
  message += "\n";

  threshold_t = ((double)getTickCount() - threshold_t)/getTickFrequency();

  // Show contours in a window
  namedWindow("Shapes", CV_WINDOW_AUTOSIZE);
  imshow("Shapes", drawing);
  //printf("\rImage FPS: %lf \t Contour FPS: %lf", 1/image_t, 1/threshold_t);

}

void denoiseMat(Mat frame, int structure_size) {
  //morphological opening (remove small objects from the foreground)
  erode(frame, frame, getStructuringElement(MORPH_RECT, Size(structure_size, structure_size)));
  dilate(frame, frame, getStructuringElement(MORPH_RECT, Size(structure_size, structure_size)));

  //morphological closing (fill small holes in the foreground)
  dilate(frame, frame, getStructuringElement(MORPH_RECT, Size(structure_size, structure_size)));
  erode(frame, frame, getStructuringElement(MORPH_RECT, Size(structure_size, structure_size)));

}

double rectangularity(vector<Point> polygon, Rect boundRect) {
  double boundRectArea = boundRect.width * boundRect.height;
  return (contourArea(polygon, false) / boundRectArea) * 100;
}

double circularity(vector<Point> polygon, float radius) {
  double boundCircleArea = PI*pow(radius, 2);
  return (contourArea(polygon, false) / boundCircleArea) * 100;
}
