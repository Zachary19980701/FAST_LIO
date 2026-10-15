#include <gtest/gtest.h>

#include "keyframe_pose_writer.h"

#include <dirent.h>
#include <limits>
#include <vector>

namespace {
class KeyframePoseWriterTest : public testing::Test {
 protected:
  void SetUp() override {
    char directory[] = "/tmp/fastlio_keyframe_poses_XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    directory_ = directory;
    map_path_ = directory_ + "/custom_map.pcd";
  }

  void TearDown() override {
    DIR* directory = opendir(directory_.c_str());
    if (!directory) return;
    while (auto* entry = readdir(directory)) {
      if (entry->d_name[0] != '.')
        std::remove((directory_ + "/" + entry->d_name).c_str());
    }
    closedir(directory);
    rmdir(directory_.c_str());
  }

  std::vector<std::vector<double>> readPoses(const std::string& path) {
    std::ifstream input(path);
    EXPECT_TRUE(input.good());
    std::vector<std::vector<double>> result;
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream row(line);
      std::vector<double> values;
      double value;
      while (row >> value) values.push_back(value);
      EXPECT_EQ(values.size(), 8u);
      result.push_back(values);
    }
    return result;
  }

  std::string directory_, map_path_;
};

TEST_F(KeyframePoseWriterTest, CheckpointAndFinalFilesPreservePoseOrderAndPrecision) {
  KeyframePoseWriter writer(map_path_);
  const Eigen::Vector3d position(1.234567890123, -2.5, 0.35);
  const Eigen::Quaterniond rotation(Eigen::AngleAxisd(1.2, Eigen::Vector3d::UnitZ()));
  const double stamp = 1720000000.1234567;
  writer.append(position, Eigen::Quaterniond(rotation.coeffs() * 2.0), stamp);
  writer.checkpoint();
  pcl::PointCloud<pcl::PointXYZI> trajectory;
  ASSERT_EQ(pcl::io::loadPCDFile(writer.trajectoryWorkingPath(), trajectory), 0);
  ASSERT_EQ(trajectory.size(), 1u);
  EXPECT_FLOAT_EQ(trajectory[0].x, position.x());
  EXPECT_FLOAT_EQ(trajectory[0].z, position.z());
  EXPECT_FLOAT_EQ(trajectory[0].intensity, 0.f);
  auto checkpoint = readPoses(writer.posesWorkingPath());
  ASSERT_EQ(checkpoint.size(), 1u);
  EXPECT_DOUBLE_EQ(checkpoint[0][0], stamp);
  EXPECT_DOUBLE_EQ(checkpoint[0][1], position.x());
  EXPECT_NEAR(checkpoint[0][6], rotation.z(), 1e-15);
  EXPECT_NEAR(checkpoint[0][7], rotation.w(), 1e-15);
  EXPECT_FALSE(std::ifstream(writer.trajectoryPath()).good());
  EXPECT_FALSE(std::ifstream(writer.posesPath()).good());

  writer.append(position + Eigen::Vector3d::UnitX(), Eigen::Quaterniond::Identity(), stamp + 1.0);
  writer.finalize();
  writer.finalize();
  EXPECT_EQ(writer.count(), 2u);
  ASSERT_EQ(pcl::io::loadPCDFile(writer.trajectoryPath(), trajectory), 0);
  ASSERT_EQ(trajectory.size(), 2u);
  EXPECT_FLOAT_EQ(trajectory[1].x, position.x() + 1.0);
  EXPECT_FLOAT_EQ(trajectory[1].intensity, 1.f);
  const auto poses = readPoses(writer.posesPath());
  ASSERT_EQ(poses.size(), 2u);
  EXPECT_DOUBLE_EQ(poses[1][0], stamp + 1.0);
  EXPECT_DOUBLE_EQ(poses[1][7], 1.0);
  EXPECT_FALSE(std::ifstream(writer.trajectoryWorkingPath()).good());
  EXPECT_FALSE(std::ifstream(writer.posesWorkingPath()).good());
  EXPECT_THROW(writer.append(position, rotation, stamp + 2.0), std::logic_error);
}

TEST_F(KeyframePoseWriterTest, EmptyRunPreservesPreviousOutputs) {
  const auto trajectory_path = directory_ + "/trajectory.pcd";
  const auto poses_path = directory_ + "/keyframe_poses.tum";
  std::ofstream(trajectory_path) << "previous trajectory";
  std::ofstream(poses_path) << "previous poses";
  KeyframePoseWriter writer(map_path_);
  writer.checkpoint();
  writer.finalize();
  std::string contents;
  std::getline(std::ifstream(trajectory_path), contents);
  EXPECT_EQ(contents, "previous trajectory");
  std::getline(std::ifstream(poses_path), contents);
  EXPECT_EQ(contents, "previous poses");
  EXPECT_FALSE(std::ifstream(writer.trajectoryWorkingPath()).good());
  EXPECT_FALSE(std::ifstream(writer.posesWorkingPath()).good());
}

TEST_F(KeyframePoseWriterTest, InvalidPosesAndConflictingPathsAreRejected) {
  KeyframePoseWriter writer(map_path_);
  const auto identity = Eigen::Quaterniond::Identity();
  const auto zero = Eigen::Vector3d::Zero();
  EXPECT_THROW(writer.append(zero, Eigen::Quaterniond(0, 0, 0, 0), 1.0), std::invalid_argument);
  EXPECT_THROW(writer.append(zero, identity, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
  EXPECT_THROW(writer.append(Eigen::Vector3d(1e100, 0, 0), identity, 1.0), std::invalid_argument);
  EXPECT_EQ(writer.count(), 0u);
  EXPECT_THROW(KeyframePoseWriter conflicting(writer.trajectoryPath()), std::invalid_argument);
  EXPECT_THROW(KeyframePoseWriter conflicting(writer.posesPath()), std::invalid_argument);
  EXPECT_FALSE(std::ifstream(writer.trajectoryWorkingPath()).good());
}

TEST_F(KeyframePoseWriterTest, ExistingPoseCheckpointIsPreserved) {
  KeyframePoseWriter writer(map_path_);
  std::ofstream(writer.posesWorkingPath()) << "recoverable checkpoint";
  EXPECT_THROW(writer.append(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), 1.0),
               std::runtime_error);
  std::string contents;
  std::getline(std::ifstream(writer.posesWorkingPath()), contents);
  EXPECT_EQ(contents, "recoverable checkpoint");
  EXPECT_EQ(writer.count(), 0u);
}
}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
