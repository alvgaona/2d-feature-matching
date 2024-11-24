#pragma once

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <vector>

#include "data_frame.h"

void DetectKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
void DetectKeypointsShiTomasi(
  std::vector<cv::KeyPoint> &keypoints, cv::Mat &img
);
void DetectKeypointsModern(
  std::vector<cv::KeyPoint> &keypoints,
  cv::Mat &img,
  const std::string &detector_type
);
void DescribeKeypoints(
  std::vector<cv::KeyPoint> &keypoints,
  cv::Mat &img,
  cv::Mat &descriptors,
  const std::string &descriptor_type
);
void MatchDescriptors(
  cv::Mat &descriptors_src,
  cv::Mat &descriptors_ref,
  std::vector<cv::DMatch> &matches,
  const std::string &matcher_type,
  const std::string &selector_type
);
