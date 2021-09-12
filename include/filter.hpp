#pragma once

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

// #define ORB_DETECTOR
#ifndef ORB_DETECTOR
#define FAST_DETECTOR
#define CORNER_THRESHOLD 20
#endif

#define DAMP_FACTOR 1
#define FILTER_SIZE 15
#define MOTION_THRESH 75
#define SCALE_FACTOR 1.2
#define HISTORY_LIMIT 30
#define SMOOTHING_RADIUS 25  // Must be less than HISTORY LIMIT

#define _DECL_STABILIZER_DEBUG
#define _DECL_STABILIZER_FPS

inline bool fKeyPointComparator(const cv::KeyPoint &p1, const cv::KeyPoint &p2);
std::vector<cv::Point2f> fKeyPoint2StdVector(std::vector<cv::KeyPoint> &keypoints, int nCorner = 300);
int fCleanPoints(std::vector<cv::Point2f> &prevPoints, std::vector<cv::Point2f> &currPoints, std::vector<uchar> &status);
cv::Point3f fMovingAverage(const std::deque<cv::Point3f> &vPointsInTime, const int &smoothRadius);
void fCropBorder(cv::UMat &frame);
int signnum_typical(double x);
