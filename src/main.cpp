#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/encoded_image.hpp>
#include <rerun/archetypes/line_strips2d.hpp>
#include <rerun/archetypes/points2d.hpp>
#include <rerun/archetypes/text_log.hpp>
#include <rerun/recording_stream.hpp>
#include <vector>

#include "collection_adapters.h"
#include "data_frame.h"
#include "frame_buffer.h"
#include "matching_2d.h"

namespace fs = std::filesystem;

int main() {
  FrameBuffer<DataFrame> frame_buffer(2);

  /* Configuration */
  std::string detector_type = "SIFT";
  std::string descriptor_type = "SIFT";
  std::string matcher_type = "MAT_FLANN";
  std::string selector_type = "SEL_KNN";

  int max_keypoints = 0;

  std::vector<fs::path> img_paths;

  const auto rec = rerun::RecordingStream("feature_tracking");
  rec.spawn().exit_on_failure();

  for (const auto& entry :
       fs::directory_iterator("./images/KITTI/2011_09_26/image_00/data")) {
    img_paths.push_back(entry.path());
  }

  std::sort(img_paths.begin(), img_paths.end());

  for (const auto& img_path : img_paths) {
    cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);

    DataFrame frame;
    frame.image = img;
    frame_buffer.append(frame);

    std::vector<cv::KeyPoint> keypoints;

    if (detector_type == "SHITOMASI") {
      DetectKeypointsShiTomasi(keypoints, img);
    } else if (detector_type == "HARRIS") {
      DetectKeypointsHarris(keypoints, img);
    } else {
      DetectKeypointsModern(keypoints, img, detector_type);
    }

    if (max_keypoints > 0) {
      // there is no response info, so keep the first
      // 50 as they are sorted in descending quality order
      if (detector_type == "SHITOMASI") {
        keypoints.erase(keypoints.begin() + max_keypoints, keypoints.end());
      }
      cv::KeyPointsFilter::retainBest(keypoints, max_keypoints);
    }

    frame_buffer.begin()->keypoints = keypoints;

    cv::Mat descriptors;

    DescribeKeypoints(
      frame_buffer.begin()->keypoints,
      frame_buffer.begin()->image,
      descriptors,
      descriptor_type
    );

    frame_buffer.begin()->descriptors = descriptors;

    if (frame_buffer.size() > 1) {
      std::vector<cv::DMatch> matches;
      MatchDescriptors(
        (frame_buffer.begin() + 1)->descriptors,
        frame_buffer.begin()->descriptors,
        matches,
        matcher_type,
        selector_type
      );

      frame_buffer.begin()->keypoints_matches = matches;

      // Logging in Rerun
      auto prev_frame = frame_buffer.at(0);
      auto curr_frame = frame_buffer.at(1);

      rec.log(
        "logs",
        rerun::TextLog(
          "Number of matches found: " + std::to_string(matches.size())
        )
          .with_level(rerun::TextLogLevel::Info)
      );

      uint32_t width = prev_frame.image.cols;
      uint32_t height = prev_frame.image.rows;

      std::vector<rerun::Position2D> points_2d_curr(matches.size());
      std::vector<rerun::Position2D> points_2d_prev(matches.size());
      std::vector<rerun::LineStrip2D> match_edges(matches.size() / 2);

      for (const auto& match : matches) {
        const auto& prev_kp = prev_frame.keypoints[match.trainIdx];
        const auto& curr_kp = curr_frame.keypoints[match.queryIdx];

        points_2d_prev.emplace_back(prev_kp.pt.x, prev_kp.pt.y);
        points_2d_curr.emplace_back(
          curr_kp.pt.x + static_cast<float>(width), curr_kp.pt.y
        );
        match_edges.push_back(rerun::LineStrip2D({
          rerun::Position2D(prev_kp.pt.x, prev_kp.pt.y),
          rerun::Position2D(
            curr_kp.pt.x + static_cast<float>(width), curr_kp.pt.y
          ),
        }));
      }

      cv::Mat concat_image;
      cv::hconcat(
        frame_buffer.begin()->image,
        (frame_buffer.begin() + 1)->image,
        concat_image
      );
      rec.log(
        "matches/image",
        rerun::Image::from_greyscale8(concat_image, {width * 2, height})
      );

      rec.log(
        "matches/image/prev/keypoints",
        rerun::Points2D(points_2d_prev).with_colors(rerun::Color(255, 0, 0))
      );
      rec.log(
        "matches/image/curr/keypoints",
        rerun::Points2D(points_2d_curr).with_colors(rerun::Color(0, 0, 255))
      );

      rec.log(
        "matches/image/edges",
        rerun::LineStrips2D(match_edges)
          .with_radii(0.1f)
          .with_colors(rerun::Color(153, 50, 204))
      );
    }
  }

  return 0;
}
