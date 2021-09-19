#pragma once

#include <iostream>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include "filter.hpp"
#include "render.hpp"

// #define DEBUG

void mainProcess(char *input, int thID, int deviceNum, bool enhance = true);