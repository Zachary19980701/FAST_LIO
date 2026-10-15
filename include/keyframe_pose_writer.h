#pragma once

#include "pcd_stream_writer.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>

#include <cmath>
#include <locale>

// Stream selected LiDAR poses without retaining a trajectory in memory.
// Final files sit beside the map; checkpoints remain recoverable until the
// map has been exported successfully and finalize() is called.
class KeyframePoseWriter {
 public:
  explicit KeyframePoseWriter(const std::string& map_path)
      : trajectory_path_(directory(map_path) + "trajectory.pcd"),
        poses_path_(directory(map_path) + "keyframe_poses.tum"),
        working_poses_path_(poses_path_ + ".in_progress." + std::to_string(getpid())),
        trajectory_(trajectory_path_) {
    if (map_path.empty() || map_path == trajectory_path_ || map_path == poses_path_)
      throw std::invalid_argument("Map path must differ from keyframe pose output paths");
  }

  void append(const Eigen::Vector3d& position, const Eigen::Quaterniond& rotation,
              double stamp) {
    if (finalized_) throw std::logic_error("Cannot append to finalized keyframe poses");
    const double norm = rotation.norm();
    if (!position.cast<float>().allFinite() || !rotation.coeffs().allFinite() ||
        !std::isfinite(norm) || norm < 1e-12 || !std::isfinite(stamp))
      throw std::invalid_argument("Invalid saved keyframe pose or timestamp");
    const Eigen::Quaterniond normalized = rotation.normalized();
    if (!poses_.is_open()) {
      if (std::ifstream(working_poses_path_).good())
        throw std::runtime_error("Existing pose checkpoint must be preserved: " + working_poses_path_);
      poses_.exceptions(std::ios::badbit | std::ios::failbit);
      poses_.imbue(std::locale::classic());
      poses_.open(working_poses_path_);
      poses_ << "# frame_id: camera_init; child_frame_id: lidar\n"
             << "# timestamp tx ty tz qx qy qz qw\n" << std::setprecision(17);
    }

    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::PointXYZI point;
    point.x = position.x();
    point.y = position.y();
    point.z = position.z();
    point.intensity = static_cast<float>(count());  // Zero-based keyframe ID.
    cloud.push_back(point);
    trajectory_.append(cloud);
    poses_ << stamp << ' ' << position.x() << ' ' << position.y() << ' ' << position.z()
           << ' ' << normalized.x() << ' ' << normalized.y() << ' ' << normalized.z()
           << ' ' << normalized.w() << '\n';
    poses_.flush();
  }

  void checkpoint() {
    if (!count() || finalized_) return;
    trajectory_.checkpoint();
    poses_.flush();
  }

  void finalize() {
    if (!count() || finalized_) return;  // Preserve previous outputs on an empty run.
    checkpoint();
    poses_.close();
    trajectory_.finalize();
    if (std::rename(working_poses_path_.c_str(), poses_path_.c_str()) != 0)
      throw std::runtime_error("Cannot rename completed keyframe poses to " + poses_path_);
    finalized_ = true;
  }

  uint64_t count() const { return trajectory_.count(); }
  const std::string& trajectoryPath() const { return trajectory_path_; }
  const std::string& posesPath() const { return poses_path_; }
  const std::string& trajectoryWorkingPath() const { return trajectory_.workingPath(); }
  const std::string& posesWorkingPath() const { return working_poses_path_; }

 private:
  static std::string directory(const std::string& path) {
    const auto slash = path.find_last_of('/');
    return slash == std::string::npos ? "" : path.substr(0, slash + 1);
  }

  std::string trajectory_path_, poses_path_, working_poses_path_;
  PcdStreamWriter<pcl::PointXYZI> trajectory_;
  std::ofstream poses_;
  bool finalized_ = false;
};
