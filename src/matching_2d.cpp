#include "matching_2d.h"

void MatchDescriptors(
  cv::Mat &descriptors_src,
  cv::Mat &descriptors_ref,
  std::vector<cv::DMatch> &matches,
  const std::string &matcher_type,
  const std::string &selector_type
) {
  bool cross_check = false;
  cv::Ptr<cv::DescriptorMatcher> matcher;
  if (matcher_type == "MAT_BF") {
    matcher = cv::BFMatcher::create(cv::NORM_HAMMING, cross_check);
  } else if (matcher_type == "MAT_FLANN") {
    if (descriptors_src.type() != CV_32F) {
      // OpenCV bug workaround : convert binary descriptors to floating point
      // due to a bug in current OpenCV implementation
      descriptors_src.convertTo(descriptors_src, CV_32F);
      descriptors_ref.convertTo(descriptors_ref, CV_32F);
    }
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  }

  if (selector_type == "SEL_NN") {
    matcher->match(descriptors_src, descriptors_ref, matches);
  } else if (selector_type == "SEL_KNN") {
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors_src, descriptors_ref, knn_matches, 2);

    double min_distance_ratio = 0.8;
    for (auto &knn_match : knn_matches) {
      if (knn_match[0].distance < min_distance_ratio * knn_match[1].distance) {
        matches.push_back(knn_match[0]);
      }
    }
  }
}

void DescribeKeypoints(
  std::vector<cv::KeyPoint> &keypoints,
  cv::Mat &img,
  cv::Mat &descriptors,
  const std::string &descriptor_type
) {
  cv::Ptr<cv::DescriptorExtractor> extractor;
  if (descriptor_type == "BRISK") {
    extractor = cv::BRISK::create(30, 3, 1.0f);
  } else if (descriptor_type == "BRIEF") {
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, false);
  } else if (descriptor_type == "ORB") {
    extractor =
      cv::ORB::create(500, 1.2F, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
  } else if (descriptor_type == "FREAK") {
    extractor = cv::xfeatures2d::FREAK::create(true, true, 22.0f, 4);
  } else if (descriptor_type == "AKAZE") {
    extractor = cv::AKAZE::create(
      cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001F, 4, 4, cv::KAZE::DIFF_PM_G2
    );
  } else if (descriptor_type == "SIFT") {
    extractor = cv::SIFT::create(0, 3, 0.04, 10, 1.6);
  } else {
    throw std::runtime_error(
      "The provided descriptor algorithm is not supported."
    );
  }

  extractor->compute(img, keypoints, descriptors);
}

void DetectKeypointsShiTomasi(
  std::vector<cv::KeyPoint> &keypoints, cv::Mat &img
) {
  int block_size = 4;
  double max_overlap = 0.0;
  double quality_level = 0.01;
  double k = 0.04;

  auto min_distance = (1.0 - max_overlap) * block_size;
  int max_corners =
    static_cast<int>(img.rows * img.cols / std::max(1.0, min_distance));

  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(
    img,
    corners,
    max_corners,
    quality_level,
    min_distance,
    cv::Mat(),
    block_size,
    false,
    k
  );

  // add corners to result vector
  for (auto &corner : corners) {
    cv::KeyPoint keypoint;
    keypoint.pt = cv::Point2f(corner.x, corner.y);
    keypoint.size = static_cast<float>(block_size);
    keypoints.push_back(keypoint);
  }
}

void DetectKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {
  const int block_size = 4;
  const int aperture_size = 3;
  const double k = 0.04;
  cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
  cv::cornerHarris(img, dst, block_size, aperture_size, k);
  cv::Mat dst_norm;
  cv::Mat dst_norm_scaled;
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  convertScaleAbs(dst_norm, dst_norm_scaled);

  for (auto r = 0; r < dst_norm.rows; r++) {
    for (auto c = 0; c < dst_norm.cols; c++) {
      auto response = dst_norm.at<float>(r, c);
      const double min_response = 100;
      if (response <= min_response) {
        continue;
      }
      cv::KeyPoint kp;
      kp.pt = cv::Point2f(static_cast<float>(c), static_cast<float>(r));
      kp.size = 2 * aperture_size;
      kp.response = response;
      bool overlap = false;
      const double max_overlap = 0;
      for (auto &keypoint : keypoints) {
        double keypoint_overlap = cv::KeyPoint::overlap(kp, keypoint);
        if (keypoint_overlap > max_overlap) {
          overlap = true;
          if (kp.response > keypoint.response) {
            keypoint = kp;
            break;
          }
        }
      }
      if (!overlap) {
        keypoints.push_back(kp);
      }
    }
  }
}

void DetectKeypointsModern(
  std::vector<cv::KeyPoint> &keypoints,
  cv::Mat &img,
  const std::string &detector_type
) {
  cv::Ptr<cv::FeatureDetector> detector;
  if (detector_type == "FAST") {
    detector = cv::FastFeatureDetector::create(
      30, true, cv::FastFeatureDetector::TYPE_9_16
    );
  } else if (detector_type == "BRISK") {
    detector = cv::BRISK::create(30, 3, 1.0);
  } else if (detector_type == "ORB") {
    detector =
      cv::ORB::create(750, 1.2F, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
  } else if (detector_type == "AKAZE") {
    detector = cv::AKAZE::create(
      cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001F, 5, 5, cv::KAZE::DIFF_PM_G2
    );
  } else if (detector_type == "SIFT") {
    detector = cv::SIFT::create(0, 3, 0.05, 12, 1.8);
  } else {
    throw std::runtime_error("The provided detector algorithm is not supported."
    );
  }

  detector->detect(img, keypoints);
}
