#pragma once

#include <deque>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>

#define MOTION_THRESH 75
#define SCALE_FACTOR 1.2
#define HISTORY_LIMIT 30
#define SMOOTHING_RADIUS 25 // Must be less than HISTORY LIMIT

inline bool fKeyPointComparator(const cv::KeyPoint &p1, const cv::KeyPoint &p2);
std::vector<cv::Point2f> fKeyPoint2StdVector(std::vector<cv::KeyPoint> &keypoints, int nCorner = 300);
int fCleanPoints(cv::Mat &prevPoints, cv::Mat &currPoints, cv::Mat &status);
cv::Point3f fMovingAverage(const std::deque<cv::Point3f> &vPointsInTime, const int &smoothRadius);
int signnum_typical(double x);
