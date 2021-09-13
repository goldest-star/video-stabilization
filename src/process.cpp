#include "process.hpp"

void mainProcess(char *input, char *output, int deviceNum, bool enhance)
{
    cv::cuda::setDevice(deviceNum);

    // Init variables
    int camVal = 0;
    bool showFlag = false;
    char *strPtr = nullptr;
    std::vector<uchar> status;
    std::vector<cv::KeyPoint> vKeyPoints;
    std::vector<cv::Point2f> prevPoints, currPoints;
    std::deque<cv::Point3f> vPointsInTime(HISTORY_LIMIT);

    double kernelArr[] = {0, -1, 0, -1, 5, -1, 0, -1, 0}; // Unsharp masking
    cv::Mat kernel(3, 3, CV_64F, kernelArr), H, cropperMat;

    cv::Mat orgCPUFrame, outCPUFrame;
    cv::cuda::GpuMat orgFrame, currFrame, prevFrame, bufferFrame;
    std::vector<cv::cuda::GpuMat> splitFrame;

    cv::Ptr<cv::cuda::ORB> pDetector;
    cv::Ptr<cv::cuda::Filter> pFilter;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> pTracker;

    cv::VideoCapture capVid;
    cv::VideoWriter writerVid;

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

    // Init output engine;
    camVal = strtol(output, &strPtr, 10);
    if (strPtr - output != strlen(output))
    {
        cv::namedWindow(input, cv::WINDOW_OPENGL);
        showFlag = true;
    }
    else
        writerVid.open(output,
                       cv::CAP_FFMPEG,
                       capVid.get(cv::CAP_PROP_FOURCC),
                       capVid.get(cv::CAP_PROP_FPS),
                       cv::Size(capVid.get(cv::CAP_PROP_FRAME_WIDTH), capVid.get(cv::CAP_PROP_FRAME_HEIGHT)),
                       capVid.get(cv::CAP_PROP_CHANNEL) - 1);

    if (!showFlag && !writerVid.isOpened())
    {
        printf("Can't init output %s\n", output);
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

    orgFrame.upload(orgCPUFrame);
    cv::cuda::cvtColor(orgFrame, prevFrame, cv::COLOR_BGR2GRAY);
    pDetector->detect(prevFrame, vKeyPoints);
    prevPoints = fKeyPoint2StdVector(vKeyPoints);

    // Init filter engine
    if (enhance)
    {
        pFilter = cv::cuda::createLinearFilter(orgFrame.type(), orgFrame.type(), kernel);
        if (pFilter.empty())
        {
            printf("Can't init filter engine\n");
            goto cleanup;
        }
    }

    // Main process
    cropperMat = cv::getRotationMatrix2D(cv::Point2f(orgFrame.cols / 2, orgFrame.rows / 2), 0, SCALE_FACTOR);
    while (capVid.isOpened())
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
        H = cv::estimateAffine2D(prevPoints, currPoints);
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
            pFilter->apply(orgFrame, orgFrame);
            cv::cuda::cvtColor(orgFrame, bufferFrame, cv::COLOR_BGR2HSV);
            cv::cuda::split(bufferFrame, splitFrame);
            cv::cuda::equalizeHist(splitFrame[1], splitFrame[1]);
            cv::cuda::merge(splitFrame, bufferFrame);
            cv::cuda::cvtColor(bufferFrame, orgFrame, cv::COLOR_HSV2BGR);
            cv::cuda::bilateralFilter(orgFrame, orgFrame, 5, 45, 45);
        }

        // Stabilize
        cv::cuda::warpAffine(orgFrame, orgFrame, H, orgFrame.size(), cv::WARP_INVERSE_MAP);

        // Crop
        cv::cuda::warpAffine(orgFrame, orgFrame, cropperMat, orgFrame.size());

        // Write output
        orgFrame.download(outCPUFrame);
        if (showFlag)
        {
            cv::imshow(input, outCPUFrame);
            cv::waitKey(1);
        }
        else
            writerVid.write(outCPUFrame);

        // Swap frames
        cv::cuda::swap(prevFrame, currFrame);

        // Calculate points
        pDetector->detect(prevFrame, vKeyPoints);
        prevPoints = fKeyPoint2StdVector(vKeyPoints);
    }

cleanup:
    printf("Closing\n");

    capVid.release();
    if (showFlag)
        cv::destroyWindow(input);
    else
        writerVid.release();

    return;
}
