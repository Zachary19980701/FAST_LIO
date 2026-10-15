#include <gtest/gtest.h>
#include "map_save_keyframe.h"

TEST(MapSaveKeyframe, StationaryScansDoNotBecomePeriodicKeyframes) {
  MapSaveKeyframe selector(MapSaveKeyframe::Options{});
  EXPECT_TRUE(selector.select(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), 1.0));
  for (int i = 1; i <= 1000; ++i)
    EXPECT_FALSE(selector.select(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), 1.0 + i));
}

TEST(MapSaveKeyframe, TranslationAccumulatesSinceLastSavedScan) {
  MapSaveKeyframe selector(MapSaveKeyframe::Options{});
  const auto rotation = Eigen::Quaterniond::Identity();
  EXPECT_TRUE(selector.select(Eigen::Vector3d::Zero(), rotation, 0.0));
  EXPECT_FALSE(selector.select(Eigen::Vector3d(0.4, 0, 0), rotation, 0.1));
  EXPECT_TRUE(selector.select(Eigen::Vector3d(0.8, 0, 0), rotation, 0.2));
  EXPECT_FALSE(selector.select(Eigen::Vector3d(0.9, 0, 0), rotation, 0.3));
}

TEST(MapSaveKeyframe, SlowMotionHasATimeLimit) {
  MapSaveKeyframe selector(MapSaveKeyframe::Options{});
  const auto rotation = Eigen::Quaterniond::Identity();
  EXPECT_TRUE(selector.select(Eigen::Vector3d::Zero(), rotation, 0.0));
  EXPECT_FALSE(selector.select(Eigen::Vector3d(0.2, 0, 0), rotation, 1.9));
  EXPECT_TRUE(selector.select(Eigen::Vector3d(0.2, 0, 0), rotation, 2.0));
  EXPECT_FALSE(selector.select(Eigen::Vector3d(0.205, 0, 0), rotation, 20.0));
}

TEST(MapSaveKeyframe, PureRotationAndQuaternionSignAreHandled) {
  MapSaveKeyframe selector(MapSaveKeyframe::Options{});
  EXPECT_TRUE(selector.select(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), 0.0));
  Eigen::Quaterniond rotation(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()));
  EXPECT_TRUE(selector.select(Eigen::Vector3d::Zero(), rotation, 0.1));
  rotation.coeffs() *= -1.0;
  EXPECT_FALSE(selector.select(Eigen::Vector3d::Zero(), rotation, 0.2));
}

TEST(MapSaveKeyframe, InvalidThresholdsAndPosesAreRejected) {
  MapSaveKeyframe::Options options;
  options.translation = 0.0;
  EXPECT_THROW(MapSaveKeyframe selector(options), std::invalid_argument);
  options.translation = 0.75;
  options.rotation_deg = 181.0;
  EXPECT_THROW(MapSaveKeyframe selector(options), std::invalid_argument);
  MapSaveKeyframe selector(MapSaveKeyframe::Options{});
  EXPECT_THROW(selector.select(Eigen::Vector3d::Zero(), Eigen::Quaterniond(0, 0, 0, 0), 0.0),
               std::invalid_argument);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
