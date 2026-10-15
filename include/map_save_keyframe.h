#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Geometry>

// Select saved scans independently of the per-scan state estimator and local map.
class MapSaveKeyframe {
 public:
  struct Options {
    double translation = 0.75;
    double rotation_deg = 10.0;
    double max_interval = 2.0;
  };

  explicit MapSaveKeyframe(const Options& options) : options_(options) {
    if (!std::isfinite(options.translation) || options.translation <= 0.0 ||
        !std::isfinite(options.rotation_deg) || options.rotation_deg <= 0.0 ||
        options.rotation_deg > 180.0 || !std::isfinite(options.max_interval) ||
        options.max_interval < 0.0)
      throw std::invalid_argument("Invalid map-save keyframe thresholds");
  }

  bool select(const Eigen::Vector3d& position, const Eigen::Quaterniond& rotation,
              double stamp) {
    if (!position.allFinite() || !rotation.coeffs().allFinite() ||
        rotation.norm() < 1e-12 || !std::isfinite(stamp))
      throw std::invalid_argument("Invalid map-save keyframe pose or timestamp");
    const Eigen::Quaterniond normalized = rotation.normalized();
    bool selected = !initialized_;
    if (initialized_) {
      const double distance = (position - position_).norm();
      const double angle = rotation_.angularDistance(normalized) * 180.0 / std::acos(-1.0);
      // A time limit fills gaps during slow motion, without saving a stationary
      // scanner repeatedly. Smaller configured thresholds still take precedence.
      const bool moving = distance >= std::min(0.1, options_.translation) ||
                          angle >= std::min(2.0, options_.rotation_deg);
      selected = distance >= options_.translation || angle >= options_.rotation_deg ||
                 (options_.max_interval > 0.0 && stamp - stamp_ >= options_.max_interval && moving);
    }
    if (selected) {
      initialized_ = true;
      position_ = position;
      rotation_ = normalized;
      stamp_ = stamp;
    }
    return selected;
  }

 private:
  Options options_;
  bool initialized_ = false;
  double stamp_ = 0.0;
  Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rotation_ = Eigen::Quaterniond::Identity();
};
