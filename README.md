# 2D Feature Tracking

![Feature matching with Rerun.io](img/feature-matching-rerun.mp4)

This project demonstrates 2D feature matching using a sequence of images from the KITTI
dataset. Multiple combinations of keypoint detectors (like FAST, BRISK, ORB, AKAZE,
SIFT) and descriptor extractors (like BRIEF, ORB, FREAK, AKAZE, SIFT) are evaluated
to find optimal pairings for feature tracking. The implementation processes a reduced
set of consecutive frames to analyze detection, description and matching performance
across different methods.

The project is integrated with Rerun.io, a visualization toolkit
that enables real-time inspection of the feature detection and matching process, making
it easier to debug and understand the behavior of different detector/descriptor combinations.
