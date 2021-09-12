#include "filter.hpp"

bool fKeyPointComparator(const cv::KeyPoint &p1, const cv::KeyPoint &p2) { return p1.response > p2.response; }

std::vector<cv::Point2f> fKeyPoint2StdVector(std::vector<cv::KeyPoint> &keypoints, int nCorner) {
  std::vector<cv::Point2f> pts;

  std::sort(keypoints.begin(), keypoints.end(), fKeyPointComparator);

  int i = 0;
  for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
    ++i;
    pts.push_back(it->pt);
    if (i >= nCorner) break;
  }
  return pts;
}

int fCleanPoints(std::vector<cv::Point2f> &prevPoints, std::vector<cv::Point2f> &currPoints, std::vector<uchar> &status) {
  int N = 0;
  std::vector<cv::Point2f> newPrevPoints, newCurrPoints;

  newPrevPoints.reserve(status.size());
  newCurrPoints.reserve(status.size());

  for (int idx = 0; idx < status.size(); ++idx) {
    if (status[idx]) {
      newPrevPoints.push_back(prevPoints[idx]);
      newCurrPoints.push_back(currPoints[idx]);
      ++N;
    }
  }
  return N;
}

cv::Point3f fMovingAverage(const std::deque<cv::Point3f> &vPointsInTime, const int &smoothRadius) {
  cv::Point3f output(0, 0, 0);
  std::vector<cv::Point3f> vSum;

  vSum.reserve(smoothRadius);
  for (int idx = vPointsInTime.size() - smoothRadius; idx < vPointsInTime.size(); ++idx) {
    cv::Point3f sum(0, 0, 0);
    for (int jdx = 0; jdx < idx; ++jdx) {
      sum += vPointsInTime[jdx];
    }
    vSum.push_back(sum);
  }

  for (int idx = 0; idx < vSum.size(); ++idx) {
    output += vSum[idx];
  }

  return cv::Point3f(output.x / vSum.size(), output.y / vSum.size(), output.z / vSum.size()) - vSum[vSum.size() - 1];
}

void fCropBorder(cv::UMat &frame) {
  cv::UMat frame2;
  cv::Mat H = cv::getRotationMatrix2D(cv::Point2f(frame.cols / 2, frame.rows / 2), 0, SCALE_FACTOR);
  cv::warpAffine(frame, frame2, H, frame.size());

  cv::swap(frame, frame2);
}

int signnum_typical(double x) {
  if (x > 0.0) return 1;
  if (x < 0.0) return -1;
  return 0;
}