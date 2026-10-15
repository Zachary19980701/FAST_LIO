#include <gtest/gtest.h>
#include <array>
#include <limits>

#include "tiled_voxel_map_writer.h"

namespace {
using Writer = TiledVoxelMapWriter;
using Cloud = Writer::Cloud;

Cloud::Ptr cloud(std::initializer_list<std::array<float, 4>> points) {
  Cloud::Ptr result(new Cloud);
  for (const auto& values : points) {
    Writer::Point point;
    point.x = values[0]; point.y = values[1]; point.z = values[2]; point.intensity = values[3];
    result->push_back(point);
  }
  return result;
}

void removeDirectory(const std::string& path) {
  DIR* directory = ::opendir(path.c_str());
  if (!directory) return;
  while (dirent* entry = ::readdir(directory)) {
    const std::string name(entry->d_name);
    if (name == "." || name == "..") continue;
    const std::string child = path + "/" + name;
    struct stat info;
    if (::lstat(child.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) removeDirectory(child);
    else std::remove(child.c_str());
  }
  ::closedir(directory);
  ::rmdir(path.c_str());
}

class TiledVoxelMapWriterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char path[] = "/tmp/fastlio_tiled_voxels_XXXXXX";
    ASSERT_NE(::mkdtemp(path), nullptr);
    directory_ = path;
    output_ = directory_ + "/map.pcd";
  }
  void TearDown() override { removeDirectory(directory_); }
  Writer::Options options() {
    Writer::Options result;
    result.voxel_size = 1;
    result.tile_size = 2;
    result.max_active_tiles = 1;
    result.max_queued_points = 3;
    return result;
  }
  std::string directory_, output_;
};
}  // namespace

TEST_F(TiledVoxelMapWriterTest, DeduplicatesAcrossTilesEvictionsAndNegativeCoordinates) {
  Writer writer(output_, options());
  const auto points = cloud({{.1f, .1f, .1f, 1}, {-.1f, .1f, .1f, 2},
                             {-2.1f, .1f, .1f, 3}, {2.1f, .1f, .1f, 4}});
  writer.append(points);
  writer.checkpoint();
  writer.append(points);
  writer.finalize();
  const auto stats = writer.stats();
  EXPECT_EQ(stats.received_points, 8u);
  EXPECT_EQ(stats.unique_points, 4u);
  EXPECT_EQ(stats.queued_points, 0u);
  EXPECT_LE(stats.peak_queued_points, 3u);
  EXPECT_LE(stats.active_tiles, 1u);
  EXPECT_LE(stats.peak_cache_points, 1u);
  Cloud loaded;
  ASSERT_EQ(pcl::io::loadPCDFile(output_, loaded), 0);
  EXPECT_EQ(loaded.size(), 4u);
  EXPECT_EQ(writer.workingPath().find(output_ + ".tiles."), 0u);
}

TEST_F(TiledVoxelMapWriterTest, KeepsMeasuredPointClosestToCenterAfterReload) {
  Writer writer(output_, options());
  writer.append(cloud({{.01f, .02f, .03f, 1}}));
  writer.checkpoint();
  writer.append(cloud({{2.5f, .5f, .5f, 2}}));
  writer.checkpoint();  // Evicts tile zero.
  writer.append(cloud({{.5f, .5f, .5f, 9}, {.9f, .9f, .9f, 3}}));
  writer.finalize();
  Cloud loaded;
  ASSERT_EQ(pcl::io::loadPCDFile(output_, loaded), 0);
  ASSERT_EQ(loaded.size(), 2u);
  bool found = false;
  for (const auto& point : loaded) {
    if (point.x < 1) {
      found = true;
      EXPECT_FLOAT_EQ(point.x, .5f);
      EXPECT_FLOAT_EQ(point.y, .5f);
      EXPECT_FLOAT_EQ(point.z, .5f);
      EXPECT_FLOAT_EQ(point.intensity, 9);
    }
  }
  EXPECT_TRUE(found);
  writer.finalize();  // Idempotent.
  EXPECT_THROW(writer.append(cloud({{0, 0, 0, 0}})), std::logic_error);
}

TEST_F(TiledVoxelMapWriterTest, NormalizesTileSizeAndBoundsLargeInputFrames) {
  auto config = options();
  config.voxel_size = .25;
  config.tile_size = .6;
  config.max_queued_points = 2;
  Writer writer(output_, config);
  EXPECT_DOUBLE_EQ(writer.normalizedTileSize(), .75);
  Cloud::Ptr points(new Cloud);
  for (int i = 0; i < 37; ++i) {
    Writer::Point point;
    point.x = i * .5f; point.y = -.1f; point.z = -.1f; point.intensity = i;
    points->push_back(point);
  }
  writer.append(points);  // Larger than queue limit: split rather than deadlock.
  writer.finalize();
  const auto stats = writer.stats();
  EXPECT_EQ(stats.received_points, 37u);
  EXPECT_EQ(stats.unique_points, 37u);
  EXPECT_LE(stats.peak_queued_points, 2u);
  EXPECT_EQ(stats.queued_points, 0u);
}

TEST_F(TiledVoxelMapWriterTest, CheckpointCanBeRecoveredAndPreservesExistingFinalMap) {
  std::ofstream(output_) << "old map";
  std::string checkpoint;
  {
    Writer writer(output_, options());
    writer.append(cloud({{.1f, .1f, .1f, 1}, {2.1f, .1f, .1f, 2}}));
    checkpoint = writer.checkpoint();
    std::string old;
    std::getline(std::ifstream(output_), old);
    EXPECT_EQ(old, "old map");
  }  // No finalization; tile checkpoints remain recoverable.
  EXPECT_EQ(Writer::exportCheckpoint(checkpoint, directory_ + "/recovered.pcd"), 2u);
  Cloud loaded;
  ASSERT_EQ(pcl::io::loadPCDFile(directory_ + "/recovered.pcd", loaded), 0);
  EXPECT_EQ(loaded.size(), 2u);
  std::string old;
  std::getline(std::ifstream(output_), old);
  EXPECT_EQ(old, "old map");
  EXPECT_THROW(Writer duplicate(output_, options()), std::runtime_error);
}

TEST_F(TiledVoxelMapWriterTest, WorkerFailureWakesProducerAndPropagatesWithoutOverwritingMap) {
  std::ofstream(output_) << "old map";
  auto config = options();
  config.max_queued_points = 1;
  Writer writer(output_, config);
  // A directory at the tile filename is an intentional, deterministic I/O error.
  ASSERT_EQ(::mkdir((writer.workingPath() + "/tile_0_0_0.pcd").c_str(), 0755), 0);
  EXPECT_THROW(writer.append(cloud({{.1f, .1f, .1f, 1}, {2.1f, .1f, .1f, 2}})),
               std::runtime_error);
  EXPECT_THROW(writer.checkpoint(), std::runtime_error);
  EXPECT_THROW(writer.finalize(), std::runtime_error);
  EXPECT_TRUE(writer.stats().failed);
  std::string old;
  std::getline(std::ifstream(output_), old);
  EXPECT_EQ(old, "old map");
}

TEST_F(TiledVoxelMapWriterTest, FailedFinalExportPreservesOldMapAndRecoverableTiles) {
  std::ofstream(output_) << "old map";
  Writer writer(output_, options());
  writer.append(cloud({{.5f, .5f, .5f, 1}}));
  writer.checkpoint();
  const std::string partial = output_ + ".in_progress." + std::to_string(getpid()) + ".pcd";
  std::ofstream(partial) << "previous unfinished export";
  EXPECT_THROW(writer.finalize(), std::runtime_error);
  std::string old;
  std::getline(std::ifstream(output_), old);
  EXPECT_EQ(old, "old map");
  EXPECT_EQ(Writer::exportCheckpoint(writer.workingPath(), directory_ + "/recovered.pcd"), 1u);
}

TEST_F(TiledVoxelMapWriterTest, RecoveryRejectsMissingTilesInsteadOfPublishingPartialMap) {
  std::string checkpoint;
  {
    Writer writer(output_, options());
    writer.append(cloud({{.5f, .5f, .5f, 1}, {2.5f, .5f, .5f, 2}}));
    checkpoint = writer.checkpoint();
  }
  ASSERT_EQ(std::remove((checkpoint + "/tile_0_0_0.pcd").c_str()), 0);
  std::ofstream(output_) << "old map";
  EXPECT_THROW(Writer::exportCheckpoint(checkpoint, output_), std::runtime_error);
  std::string old;
  std::getline(std::ifstream(output_), old);
  EXPECT_EQ(old, "old map");
}

TEST_F(TiledVoxelMapWriterTest, RecoveryKeepsTilesFlushedAfterLastExplicitCheckpoint) {
  std::string checkpoint;
  {
    auto config = options();
    config.max_queued_points = 1;
    Writer writer(output_, config);
    writer.append(cloud({{.5f, .5f, .5f, 1}}));
    checkpoint = writer.checkpoint();  // Manifest promises one point.
    ASSERT_EQ(::mkdir((checkpoint + "/tile_3_0_0.pcd").c_str(), 0755), 0);
    writer.append(cloud({{2.5f, .5f, .5f, 2}, {4.5f, .5f, .5f, 3}, {6.5f, .5f, .5f, 4}}));
    EXPECT_THROW(writer.checkpoint(), std::runtime_error);
  }
  // A non-file with a canonical tile filename signals damage; remove the
  // injected fault before recovery. Three valid evicted tiles remain on disk.
  ASSERT_EQ(::rmdir((checkpoint + "/tile_3_0_0.pcd").c_str()), 0);
  EXPECT_EQ(Writer::exportCheckpoint(checkpoint, directory_ + "/recovered.pcd"), 3u);
}

TEST_F(TiledVoxelMapWriterTest, SkipsInvalidPointsAndRejectsInvalidOptions) {
  auto config = options();
  config.max_queued_points = 0;
  EXPECT_THROW(Writer invalid(output_, config), std::invalid_argument);
  config = options();
  config.tile_size = std::numeric_limits<double>::infinity();
  EXPECT_THROW(Writer invalid(output_, config), std::invalid_argument);
  Writer writer(output_, options());
  writer.append(cloud({{0, 0, 0, 1}, {std::numeric_limits<float>::quiet_NaN(), 0, 0, 1}}));
  writer.finalize();
  EXPECT_EQ(writer.stats().received_points, 2u);
  EXPECT_EQ(writer.stats().invalid_points, 1u);
  EXPECT_EQ(writer.stats().unique_points, 1u);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
