#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include "pcd_stream_writer.h"

TEST(PcdStreamWriter, CheckpointsAndFinalMapPreserveEveryField) {
  char directory[] = "/tmp/fastlio_pcd_stream_XXXXXX";
  ASSERT_NE(mkdtemp(directory), nullptr);
  const std::string path = std::string(directory) + "/map.pcd";
  PcdStreamWriter<pcl::PointXYZINormal> writer(path);
  pcl::PointCloud<pcl::PointXYZINormal> cloud;
  for (int i = 0; i < 3; ++i) {
    pcl::PointXYZINormal point;
    point.x = i + 1; point.y = i + 2; point.z = i + 3;
    point.intensity = i + 10;
    point.normal_x = .1f; point.normal_y = .2f; point.normal_z = .3f;
    point.curvature = .4f;
    cloud.push_back(point);
  }
  writer.append(cloud);
  pcl::PointCloud<pcl::PointXYZINormal> loaded;
  ASSERT_EQ(pcl::io::loadPCDFile(writer.workingPath(), loaded), 0);
  ASSERT_EQ(loaded.size(), 3u);
  writer.append(cloud);
  writer.finalize();
  ASSERT_EQ(pcl::io::loadPCDFile(path, loaded), 0);
  ASSERT_EQ(loaded.size(), 6u);
  for (size_t i = 0; i < loaded.size(); ++i) {
    const auto& expected = cloud[i % cloud.size()];
    EXPECT_FLOAT_EQ(loaded[i].x, expected.x);
    EXPECT_FLOAT_EQ(loaded[i].y, expected.y);
    EXPECT_FLOAT_EQ(loaded[i].z, expected.z);
    EXPECT_FLOAT_EQ(loaded[i].intensity, expected.intensity);
    EXPECT_FLOAT_EQ(loaded[i].normal_x, expected.normal_x);
    EXPECT_FLOAT_EQ(loaded[i].normal_y, expected.normal_y);
    EXPECT_FLOAT_EQ(loaded[i].normal_z, expected.normal_z);
    EXPECT_FLOAT_EQ(loaded[i].curvature, expected.curvature);
  }
  EXPECT_LE(writer.bufferBytes(), 65536u);
  EXPECT_FALSE(std::ifstream(writer.workingPath()).good());
  std::remove(path.c_str());
  rmdir(directory);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(PcdStreamWriter, EmptyInputDoesNotCreateOrOverwriteMap) {
  char directory[] = "/tmp/fastlio_pcd_empty_XXXXXX";
  ASSERT_NE(mkdtemp(directory), nullptr);
  const std::string path = std::string(directory) + "/map.pcd";
  std::ofstream(path) << "existing map";
  PcdStreamWriter<pcl::PointXYZINormal> writer(path);
  writer.append(pcl::PointCloud<pcl::PointXYZINormal>());
  writer.finalize();
  std::string contents;
  std::getline(std::ifstream(path), contents);
  EXPECT_EQ(contents, "existing map");
  std::remove(path.c_str());
  rmdir(directory);
}
