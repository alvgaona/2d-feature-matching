#pragma once

#include <opencv2/core.hpp>
#include <vector>

struct DataFrame {  // NOLINT
  cv::Mat image;    // camera image

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  std::vector<cv::DMatch>
    keypoints_matches;  // Keypoints matches between previous and current frame
};
