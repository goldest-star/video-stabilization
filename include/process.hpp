#pragma once

#include "filter.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>

void mainProcess(char *input, char *output, int deviceNum, bool enhance = true);