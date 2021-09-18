#include "process.hpp"

void mainProcess(char *input, int thID, int deviceNum, bool enhance)
{
    cv::cuda::setDevice(deviceNum);

    // Init variables
    int camVal = 0;
    char *strPtr = nullptr;
    std::vector<cv::KeyPoint> vKeyPoints;
    std::deque<cv::Point3f> vPointsInTime(HISTORY_LIMIT);

    double kernelArr[] = {0, -1, 0, -1, 5, -1, 0, -1, 0}; // Unsharp masking
    cv::Mat kernel(3, 3, CV_64F, kernelArr), H, cropperMat;

    cv::Mat orgCPUFrame, outCPUFrame;
    cv::Mat prevPointBuff, currPointBuff;
    cv::cuda::GpuMat orgFrame, currFrame, prevFrame, bufferFrame, prevPoints, currPoints, status;
    std::vector<cv::cuda::GpuMat> splitFrame;

    cv::Ptr<cv::cuda::ORB> pDetector;
    cv::Ptr<cv::cuda::Filter> pFilter;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> pTracker;

    cv::VideoCapture capVid;

    // Init capture engine
    camVal = strtol(input, &strPtr, 10);
    if (strPtr - input == strlen(input))
        capVid.open(camVal);
    else
        capVid.open(input, cv::CAP_FFMPEG);
    if (!capVid.isOpened())
    {
        printf("Can't init input %s\n", input);
        goto cleanup;
    }

    // Init detection engine
    pDetector = cv::cuda::ORB::create();
    if (pDetector.empty())
    {
        printf("Can't init detection engine\n");
        goto cleanup;
    }

    // Init tracking engine
    pTracker = cv::cuda::SparsePyrLKOpticalFlow::create();
    if (pTracker.empty())
    {
        printf("Can't init tracking engine\n");
        goto cleanup;
    }

    // Read first frame and process
    if (!capVid.read(orgCPUFrame))
    {
        printf("Can't get frame\n");
        goto cleanup;
    }

    // Find initial keypoints
    while (prevPointBuff.empty())
    {
        orgFrame.upload(orgCPUFrame);
        cv::cuda::cvtColor(orgFrame, prevFrame, cv::COLOR_BGR2GRAY);
        pDetector->detect(prevFrame, vKeyPoints);
        if (vKeyPoints.size())
        {
            prevPointBuff = cv::Mat(fKeyPoint2StdVector(vKeyPoints), CV_32FC2).t();
            prevPoints.upload(prevPointBuff);
        }
    }

    // Init filter engine
    if (enhance)
    {
        pFilter = cv::cuda::createLinearFilter(CV_8U, CV_8U, kernel);
        if (pFilter.empty())
        {
            printf("Can't init filter engine\n");
            goto cleanup;
        }
    }

    // Main process
    cropperMat = cv::getRotationMatrix2D(cv::Point2f(orgFrame.cols / 2, orgFrame.rows / 2), 0, SCALE_FACTOR);
    while (loopFlag && capVid.isOpened())
    {
        // Read frame
        if (!capVid.read(orgCPUFrame))
        {
            printf("Can't get frame\n");
            break;
        }
        orgFrame.upload(orgCPUFrame);
        cv::cuda::cvtColor(orgFrame, currFrame, cv::COLOR_BGR2GRAY);

        // Calculate motion vectors
        pTracker->calc(prevFrame, currFrame, prevPoints, currPoints, status);

        // Calculate Transformation
        currPoints.download(currPointBuff);
        H = cv::estimateAffine2D(prevPointBuff, currPointBuff);
        cv::Point3f d(H.at<double>(0, 2), H.at<double>(1, 2), atan2(H.at<double>(1, 0), H.at<double>(0, 0)));

        // Update buffer
        if (vPointsInTime.size() >= HISTORY_LIMIT)
            vPointsInTime.pop_front();
        vPointsInTime.push_back(d);

        // Filter the motion
        d = d - fMovingAverage(vPointsInTime, SMOOTHING_RADIUS);
        if (abs(d.x) > MOTION_THRESH)
            d.x = MOTION_THRESH * signnum_typical(d.x);
        if (abs(d.y) > MOTION_THRESH)
            d.y = MOTION_THRESH * signnum_typical(d.y);

        float arrH[6] = {cos(d.z), -sin(d.z), d.x, sin(d.z), cos(d.z), d.y};
        H = cv::Mat(2, 3, CV_32F, arrH);

        // Enhancement
        if (enhance)
        {
            cv::cuda::bilateralFilter(orgFrame, orgFrame, 5, 45, 45);

            cv::cuda::split(orgFrame, splitFrame);
            pFilter->apply(splitFrame[0], splitFrame[0]);
            pFilter->apply(splitFrame[1], splitFrame[1]);
            pFilter->apply(splitFrame[2], splitFrame[2]);
            cv::cuda::merge(splitFrame, orgFrame);

            cv::cuda::cvtColor(orgFrame, bufferFrame, cv::COLOR_BGR2HLS);

            cv::cuda::split(bufferFrame, splitFrame);
            cv::cuda::equalizeHist(splitFrame[1], splitFrame[1]);
            cv::cuda::merge(splitFrame, bufferFrame);

            // Convert RGB since older OpenGL does not support BGR
            cv::cuda::cvtColor(bufferFrame, orgFrame, cv::COLOR_HLS2RGB);

            cv::cuda::bilateralFilter(orgFrame, orgFrame, 5, 45, 45);
        }

        // Stabilize
        cv::cuda::warpAffine(orgFrame, bufferFrame, H, orgFrame.size(), cv::WARP_INVERSE_MAP);

        // Crop
        cv::cuda::warpAffine(bufferFrame, orgFrame, cropperMat, orgFrame.size());

        // Write output
        orgFrame.download(orgCPUFrame);
        updateWindow(subWindows[thID], orgCPUFrame);

        // Swap gray frames
        cv::cuda::swap(prevFrame, currFrame);

        // Calculate points
        pDetector->detect(prevFrame, vKeyPoints);
        if (vKeyPoints.size())
        {
            prevPointBuff = cv::Mat(fKeyPoint2StdVector(vKeyPoints), CV_32FC2).t();
            prevPoints.upload(prevPointBuff);
        }
    }

cleanup:
    printf("Closing\n");
    glutDestroyWindow(subWindows[thID]);

    capVid.release();

    return;
}
