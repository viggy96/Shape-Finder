#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <thread>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace boost::asio::ip;

VideoCapture camera;
Mat frame;
bool frameExists = false;
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
int num_threads;

void thresh_callback(int, void*);
double rectangularity(vector<Point>, Rect);
double circularity(vector<Point>, float);

int main(int argc, char *argv[]) {
  //printf("CUDA: %i", getCudaEnabledDeviceCount());
  VideoCapture camera;
  if (argc > 1) {
    std::string address;
    address.append(argv[1]);
    address.append("video?x.mjpeg");
    camera = VideoCapture(address);
  }
  else camera = VideoCapture(CV_CAP_ANY);

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

    namedWindow("Image", CV_WINDOW_AUTOSIZE);
    imshow("Image", frame);
    image_t = ((double)getTickCount() - image_t)/getTickFrequency();


    createTrackbar("Canny Threshold:", "Image", &thresh, max_thresh, thresh_callback);
    createTrackbar("Canny Max", "Image", &thresh_max, max_thresh, thresh_callback);
    createTrackbar("LowH", "Image", &lowH, 179); //Hue (0 - 179)
    createTrackbar("HighH", "Image", &highH, 179);

    createTrackbar("LowS", "Image", &lowS, 255); //Saturation (0 - 255)
    createTrackbar("HighS", "Image", &highS, 255);

    createTrackbar("LowV", "Image", &lowV, 255); //Value (0 - 255)
    createTrackbar("HighV", "Image", &highV, 255);

    createTrackbar("Min Contour Size:", "Image", &minContourSize, 64);

    thresh_callback(0, 0);
  }

  waitKey(0);
  return 0;
}

/** @function thresh_callback */
void thresh_callback(int, void*) {
  Mat frame_hsv, frame_grey, frame_thresh, canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  float biggestArea = 0;

  cvtColor(frame, frame_hsv, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

  inRange(frame_hsv, Scalar(lowH, lowS, lowV), Scalar(highH, highS, highV), frame_thresh); //Threshold the image

  //morphological opening (remove small objects from the foreground)
  erode(frame_thresh, frame_thresh, getStructuringElement(MORPH_RECT, Size(3, 3)));
  dilate(frame_thresh, frame_thresh, getStructuringElement(MORPH_RECT, Size(3, 3)));

  //morphological closing (fill small holes in the foreground)
  dilate(frame_thresh, frame_thresh, getStructuringElement(MORPH_RECT, Size(3, 3)));
  erode(frame_thresh, frame_thresh, getStructuringElement(MORPH_RECT, Size(3, 3)));

  namedWindow("Thresholded Image", CV_WINDOW_AUTOSIZE);
  imshow("Thresholded Image", frame_thresh); //show the thresholded image

  // Detect edges using canny
  threshold_t = (double)getTickCount();
  Canny(frame_thresh, canny_output, thresh, thresh_max, 3);

  // Find contours
  findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

  // Draw contours
  Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3), drawing2 = Mat::zeros(canny_output.size(), CV_8UC3);

  vector<vector<Point> > contours_poly(contours.size());
  vector<Rect> boundRect(contours.size());
  vector<Point2f> centre(contours.size());
  vector<float> radius(contours.size());
  Scalar contour_colour(0, 0, 255);
  Scalar bound_colour(0, 255, 0);

  #pragma omp parallel for
  for (int i = 0; i < contours.size(); i++) {
    approxPolyDP(contours[i], contours_poly[i], 3, true);
    int horizontal, vertical;

    vector<Point> temp_contour_rect = contours[i], temp_contour_poly_rect = contours_poly[i];
    approxPolyDP(temp_contour_rect, temp_contour_poly_rect, 40, true);
    vector<Point> temp_contour_circle = contours[i], temp_contour_poly_circle = contours_poly[i];
    approxPolyDP(temp_contour_circle, temp_contour_poly_circle, 0, true);

    minEnclosingCircle((Mat)contours_poly[i], centre[i], radius[i]);
    boundRect[i] = boundingRect((Mat)contours_poly[i]);
    float aspect_ratio = boundRect[i].width / boundRect[i].height;

    bool isCircle = (circularity(temp_contour_poly_circle, radius[i]) > rectangularity(temp_contour_poly_rect, boundRect[i])) &&
      (temp_contour_poly_circle.size() > 50) || !(abs(temp_contour_poly_rect.size() - 4) <= 1);

    if (contourArea(contours_poly[i], false) >= minContourSize && isContourConvex(contours_poly[i])) {
      if (isCircle && circularity(temp_contour_poly_circle, radius[i]) >= 60) {
        drawContours(drawing, contours_poly, i, contour_colour, CV_FILLED, 8, hierarchy, 0, Point(0, 0));
        circle(drawing, centre[i], (int)radius[i], bound_colour, 2, 8, 0);
      }

      else if ((!isCircle && (rectangularity(temp_contour_poly_rect, boundRect[i]) >= 60 || aspect_ratio > 1))) {
        drawContours(drawing, contours_poly, i, contour_colour, CV_FILLED, 8, hierarchy, 0, Point(0, 0));
        rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), bound_colour, 2, 8, 0);
      }
    }
  }

  //drawContours(drawing2, contours, -1, contour_colour, 1, 8, hierarchy, 1, Point(0, 0));

  threshold_t = ((double)getTickCount() - threshold_t)/getTickFrequency();

  // Show contours in a window
  namedWindow("Contours", CV_WINDOW_AUTOSIZE);
  imshow("Contours", drawing);
  //namedWindow("Initial Contours", CV_WINDOW_AUTOSIZE);
  //imshow("Initial Contours", drawing2);
  printf("\rImage FPS: %lf \t Contour FPS: %lf", 1/image_t, 1/threshold_t);

}

double rectangularity(vector<Point> polygon, Rect boundRect) {
  double boundRectArea = boundRect.width * boundRect.height;
  return (contourArea(polygon, false) / boundRectArea) * 100;
}

double circularity(vector<Point> polygon, float radius) {
  double boundCircleArea = PI*pow(radius, 2);
  return (contourArea(polygon, false) / boundCircleArea) * 100;
}
