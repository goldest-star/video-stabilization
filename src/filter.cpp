#include "filter.hpp"

bool fKeyPointComparator(const cv::KeyPoint &p1, const cv::KeyPoint &p2) { return p1.response > p2.response; }

std::vector<cv::Point2f> fKeyPoint2StdVector(std::vector<cv::KeyPoint> &keypoints, int nCorner)
{
  std::vector<cv::Point2f> pts;

  std::sort(keypoints.begin(), keypoints.end(), fKeyPointComparator);

  int i = 0;
  for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
  {
    ++i;
    pts.push_back(it->pt);
    if (i >= nCorner)
      break;
  }
  return pts;
}

int fCleanPoints(cv::Mat &prevPoints, cv::Mat &currPoints, cv::Mat &status)
{
  int N = 0;
  std::vector<cv::Point2f> newPrevPoints, newCurrPoints;
  std::vector<cv::Point2f> prevPointsBuffer, currPointsBuffer;

  newPrevPoints.reserve(status.rows * status.cols);
  newCurrPoints.reserve(status.rows * status.cols);

  prevPointsBuffer.assign(prevPoints.begin<cv::Point2f>(), prevPoints.end<cv::Point2f>());
  currPointsBuffer.assign(currPoints.begin<cv::Point2f>(), currPoints.end<cv::Point2f>());

  for (int idx = 0; idx < status.cols * status.rows; ++idx)
  {
    if (status.data[idx])
    {
      newPrevPoints.push_back(prevPointsBuffer[idx]);
      newCurrPoints.push_back(currPointsBuffer[idx]);
      ++N;
    }
  }

  prevPoints = cv::Mat(newPrevPoints, true);
  currPoints = cv::Mat(newCurrPoints, true);
  return N;
}

cv::Point3f fMovingAverage(const std::deque<cv::Point3f> &vPointsInTime, const int &smoothRadius)
{
  cv::Point3f output(0, 0, 0);
  std::vector<cv::Point3f> vSum;

  vSum.reserve(smoothRadius);
  for (size_t idx = vPointsInTime.size() - smoothRadius; idx < vPointsInTime.size(); ++idx)
  {
    cv::Point3f sum(0, 0, 0);
    for (size_t jdx = 0; jdx < idx; ++jdx)
    {
      sum += vPointsInTime[jdx];
    }
    vSum.push_back(sum);
  }

  for (size_t idx = 0; idx < vSum.size(); ++idx)
  {
    output += vSum[idx];
  }

  return cv::Point3f(output.x / vSum.size(), output.y / vSum.size(), output.z / vSum.size()) - vSum[vSum.size() - 1];
}

int signnum_typical(double x)
{
  if (x > 0.0)
    return 1;
  if (x < 0.0)
    return -1;
  return 0;
}