#pragma once

#include "pcd_stream_writer.h"

#include <pcl/point_types.h>

#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <dirent.h>
#include <exception>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>

// A bounded, asynchronous global map store. Only the active tile cache and a
// bounded number of outstanding input points live in RAM. Every voxel keeps a
// measured point, chosen by distance to its center. Evicted tiles are reloaded
// before modification, so revisits do not grow the map with duplicate voxels.
// append/checkpoint/finalize are intended for one producer. stats() is safe from
// any thread. Disk failures propagate through append/checkpoint/finalize.
class TiledVoxelMapWriter {
 public:
  using Point = pcl::PointXYZI;
  using Cloud = pcl::PointCloud<Point>;

  struct Options {
    double voxel_size = 0.25;
    double tile_size = 10.0;
    size_t max_active_tiles = 16;
    size_t max_queued_points = 200000;
  };

  struct Stats {
    uint64_t received_points = 0;  // Enqueued points, including invalid input.
    uint64_t unique_points = 0;    // Finite voxels processed so far.
    uint64_t invalid_points = 0;
    size_t queued_points = 0;     // Includes the batch currently being processed.
    size_t peak_queued_points = 0;
    size_t active_tiles = 0;
    size_t cache_points = 0;
    size_t peak_cache_points = 0;
    uint64_t backpressure_waits = 0;
    bool failed = false;
  };

  explicit TiledVoxelMapWriter(const std::string& path)
      : TiledVoxelMapWriter(path, Options()) {}

  TiledVoxelMapWriter(const std::string& path, const Options& options)
      : final_path_(path), working_path_(path + ".tiles." + std::to_string(getpid())),
        options_(options) {
    if (path.empty() || !std::isfinite(options_.voxel_size) || options_.voxel_size <= 0 ||
        !std::isfinite(options_.tile_size) || options_.tile_size <= 0 ||
        options_.max_active_tiles == 0 || options_.max_queued_points == 0)
      throw std::invalid_argument("Invalid tiled voxel map options");
    // Integral voxel counts make tile boundaries coincide exactly with voxels.
    // Round upward, tolerating floating point roundoff around an integer ratio.
    const double ratio = options_.tile_size / options_.voxel_size;
    const double count = std::ceil(ratio - 1e-10 * std::max(1.0, ratio));
    if (!std::isfinite(count) || count > 1000000)
      throw std::invalid_argument("Tile size must be at most 1000000 voxels per axis");
    tile_voxels_ = std::max<int64_t>(1, static_cast<int64_t>(count));
    options_.tile_size = tile_voxels_ * options_.voxel_size;
    if (!std::isfinite(options_.tile_size))
      throw std::invalid_argument("Normalized map tile size is not finite");
    if (::mkdir(working_path_.c_str(), 0755) != 0)
      throw std::runtime_error("Cannot create map checkpoint directory (existing checkpoints are preserved): " +
                               working_path_ + ": " + std::strerror(errno));
    writeManifest();
    worker_ = std::thread(&TiledVoxelMapWriter::run, this);
  }

  ~TiledVoxelMapWriter() noexcept {
    try {
      bool needs_checkpoint;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        needs_checkpoint = !finalized_ && !failure_ && !stopping_;
      }
      if (needs_checkpoint) checkpoint();
    } catch (...) {
      // Leave all completed tile checkpoints and partial export files in place.
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    space_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  TiledVoxelMapWriter(const TiledVoxelMapWriter&) = delete;
  TiledVoxelMapWriter& operator=(const TiledVoxelMapWriter&) = delete;

  // Large frames are split into bounded batches. A full queue blocks here;
  // neither a frame nor a batch is silently discarded. The caller can release
  // its cloud immediately after append returns: queued batches own their data.
  void append(const Cloud::ConstPtr& cloud) {
    if (!cloud) throw std::invalid_argument("Cannot append a null map cloud");
    size_t first = 0;
    do {
      const size_t count = std::min(options_.max_queued_points, cloud->size() - first);
      std::unique_lock<std::mutex> lock(mutex_);
      checkUsable();
      if (!count) return;
      if (stats_.queued_points > options_.max_queued_points - count) {
        ++stats_.backpressure_waits;
        space_.wait(lock, [&] {
          return failure_ || stopping_ ||
                 stats_.queued_points <= options_.max_queued_points - count;
        });
        checkUsable();
      }
      Cloud::Ptr batch(new Cloud);
      batch->reserve(count);
      for (size_t i = 0; i < count; ++i) batch->push_back((*cloud)[first + i]);
      Job job;
      job.kind = Kind::Append;
      job.cloud = batch;
      jobs_.push_back(std::move(job));
      stats_.received_points += count;
      stats_.queued_points += count;
      stats_.peak_queued_points = std::max(stats_.peak_queued_points, stats_.queued_points);
      lock.unlock();
      ready_.notify_one();
      first += count;
    } while (first < cloud->size());
  }

  // The returned directory contains standard XYZI PCD tiles and a manifest.
  // After a successful checkpoint it is sufficient to recover the global map.
  std::string checkpoint() {
    barrier(Kind::Checkpoint);
    return working_path_;
  }

  // Flush all queued data, stream one tile at a time into the final PCD, then
  // atomically replace the previous map. Keep tiles for recovery/re-export.
  // As with PcdStreamWriter, an empty run preserves an existing final map.
  void finalize() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (failure_) std::rethrow_exception(failure_);
      if (finalized_) return;
    }
    barrier(Kind::Finalize);
    if (worker_.joinable()) worker_.join();
  }

  const std::string& workingPath() const { return working_path_; }
  double normalizedTileSize() const { return options_.tile_size; }

  Stats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

  // Recovery helper: run only after the original writer has stopped, or on a
  // copy of a completed checkpoint. No all-map point vector or tile list is used.
  static uint64_t exportCheckpoint(const std::string& directory, const std::string& output_path) {
    std::ifstream manifest(directory + "/manifest.txt");
    std::string signature;
    std::getline(manifest, signature);
    if (signature != "FAST_LIO_TILED_VOXEL_MAP 1")
      throw std::runtime_error("Missing or invalid tiled map manifest: " + directory);
    bool found_count = false;
    uint64_t expected_count = 0;
    std::string name, value;
    while (manifest >> name >> value) {
      if (name == "unique_points") {
        size_t consumed = 0;
        try {
          expected_count = std::stoull(value, &consumed);
          found_count = consumed == value.size() && !value.empty() && value[0] != '-';
        } catch (const std::exception&) {
          found_count = false;
        }
        if (!found_count) break;
      }
    }
    if (!found_count) throw std::runtime_error("Map manifest has no valid point count: " + directory);
    DIR* raw_directory = ::opendir(directory.c_str());
    if (!raw_directory) throw std::runtime_error("Cannot open map tiles: " + directory);
    std::unique_ptr<DIR, int (*)(DIR*)> entries(raw_directory, ::closedir);
    PcdStreamWriter<Point> output(output_path);
    for (;;) {
      errno = 0;
      dirent* entry = ::readdir(entries.get());
      if (!entry) {
        if (errno != 0) throw std::runtime_error("Cannot enumerate map tiles: " + directory);
        break;
      }
      const std::string name(entry->d_name);
      if (!isTileFilename(name)) continue;
      Cloud cloud;
      if (pcl::io::loadPCDFile<Point>(directory + "/" + name, cloud) != 0)
        throw std::runtime_error("Cannot recover map tile: " + directory + "/" + name);
      output.append(cloud);
    }
    // A process can die after evicting additional tiles but before the next
    // explicit checkpoint. Tiles only gain voxels, so keep that newer work;
    // never accept fewer points than the last completed checkpoint promised.
    if (output.count() < expected_count)
      throw std::runtime_error("Map tiles contain fewer points than checkpoint manifest; final map preserved: " +
                               directory);
    output.finalize();
    return output.count();
  }

 private:
  struct TileKey {
    int64_t x, y, z;
    bool operator==(const TileKey& other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };
  struct KeyHash {
    size_t operator()(const TileKey& key) const {
      size_t value = std::hash<int64_t>()(key.x);
      value ^= std::hash<int64_t>()(key.y) + 0x9e3779b9u + (value << 6) + (value >> 2);
      return value ^ (std::hash<int64_t>()(key.z) + 0x9e3779b9u + (value << 6) + (value >> 2));
    }
  };
  struct Representative {
    Point point;
    double squared_distance;
  };
  struct Tile {
    std::unordered_map<uint64_t, Representative> points;
    std::list<TileKey>::iterator lru;
    bool dirty = false;
  };
  enum class Kind { Append, Checkpoint, Finalize };
  struct Job {
    Kind kind = Kind::Append;
    Cloud::ConstPtr cloud;
    std::shared_ptr<std::promise<void>> done;
  };

  static std::string tileFilename(const TileKey& key) {
    return "tile_" + std::to_string(key.x) + "_" + std::to_string(key.y) + "_" +
           std::to_string(key.z) + ".pcd";
  }

  static bool isTileFilename(const std::string& name) {
    if (name.compare(0, 5, "tile_") != 0) return false;
    try {
      size_t first_end = 0, second_end = 0, third_end = 0;
      const std::string coordinates = name.substr(5);
      const int64_t x = std::stoll(coordinates, &first_end);
      if (coordinates.at(first_end) != '_') return false;
      const std::string second = coordinates.substr(first_end + 1);
      const int64_t y = std::stoll(second, &second_end);
      if (second.at(second_end) != '_') return false;
      const std::string third = second.substr(second_end + 1);
      const int64_t z = std::stoll(third, &third_end);
      return third.substr(third_end) == ".pcd" && tileFilename({x, y, z}) == name;
    } catch (const std::exception&) {
      return false;
    }
  }

  static int64_t floorDivide(int64_t numerator, int64_t denominator) {
    const int64_t quotient = numerator / denominator;
    return quotient - (numerator % denominator < 0 ? 1 : 0);
  }

  int64_t voxelCoordinate(float coordinate) const {
    const long double value = std::floor(static_cast<long double>(coordinate) / options_.voxel_size);
    if (value < std::numeric_limits<int64_t>::min() || value > std::numeric_limits<int64_t>::max())
      throw std::runtime_error("Map point coordinate exceeds voxel index range");
    return static_cast<int64_t>(value);
  }

  uint64_t localVoxel(const TileKey& voxel) const {
    // A signed remainder avoids overflow in tile_coordinate * tile_voxels near
    // INT64_MIN, while still producing [0, tile_voxels) for negative positions.
    auto local = [&](int64_t value) -> uint64_t {
      const int64_t remainder = value % tile_voxels_;
      return remainder < 0 ? remainder + tile_voxels_ : remainder;
    };
    return (local(voxel.z) * tile_voxels_ + local(voxel.y)) * tile_voxels_ + local(voxel.x);
  }

  double centerDistance(const Point& point, const TileKey& voxel) const {
    const double x = point.x - (static_cast<double>(voxel.x) + .5) * options_.voxel_size;
    const double y = point.y - (static_cast<double>(voxel.y) + .5) * options_.voxel_size;
    const double z = point.z - (static_cast<double>(voxel.z) + .5) * options_.voxel_size;
    return x * x + y * y + z * z;
  }

  Tile& getTile(const TileKey& key) {
    auto existing = tiles_.find(key);
    if (existing != tiles_.end()) {
      lru_.splice(lru_.begin(), lru_, existing->second.lru);
      return existing->second;
    }
    if (tiles_.size() >= options_.max_active_tiles) {
      auto victim = tiles_.find(lru_.back());
      flushTile(victim->first, victim->second);
      cache_points_ -= victim->second.points.size();
      lru_.pop_back();
      tiles_.erase(victim);
    }
    lru_.push_front(key);
    auto inserted = tiles_.emplace(key, Tile());
    Tile& tile = inserted.first->second;
    tile.lru = lru_.begin();
    const std::string path = working_path_ + "/" + tileFilename(key);
    struct stat info;
    if (::stat(path.c_str(), &info) == 0) {
      Cloud cloud;
      if (!S_ISREG(info.st_mode) || pcl::io::loadPCDFile<Point>(path, cloud) != 0)
        throw std::runtime_error("Cannot reload map tile: " + path);
      tile.points.reserve(cloud.size());
      for (const auto& point : cloud) {
        const TileKey voxel{voxelCoordinate(point.x), voxelCoordinate(point.y), voxelCoordinate(point.z)};
        const TileKey actual_tile{floorDivide(voxel.x, tile_voxels_), floorDivide(voxel.y, tile_voxels_),
                                  floorDivide(voxel.z, tile_voxels_)};
        if (!(actual_tile == key)) throw std::runtime_error("Invalid point in map tile: " + path);
        tile.points.emplace(localVoxel(voxel), Representative{point, centerDistance(point, voxel)});
      }
      cache_points_ += tile.points.size();
      peak_cache_points_ = std::max(peak_cache_points_, cache_points_);
    } else if (errno != ENOENT) {
      throw std::runtime_error("Cannot inspect map tile: " + path + ": " + std::strerror(errno));
    }
    return tile;
  }

  void process(const Cloud& cloud) {
    // Scans interleave distant tiles. Group this bounded batch first, so a tile
    // is loaded/evicted at most once per batch instead of once per input point.
    std::unordered_map<TileKey, std::vector<size_t>, KeyHash> groups;
    for (size_t index = 0; index < cloud.size(); ++index) {
      const auto& point = cloud[index];
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) ||
          !std::isfinite(point.intensity)) {
        ++invalid_points_;
        continue;
      }
      const TileKey voxel{voxelCoordinate(point.x), voxelCoordinate(point.y), voxelCoordinate(point.z)};
      const TileKey key{floorDivide(voxel.x, tile_voxels_), floorDivide(voxel.y, tile_voxels_),
                        floorDivide(voxel.z, tile_voxels_)};
      groups[key].push_back(index);
    }
    for (const auto& group : groups) {
      Tile& tile = getTile(group.first);
      for (const size_t point_index : group.second) {
        const auto& point = cloud[point_index];
        const TileKey voxel{voxelCoordinate(point.x), voxelCoordinate(point.y), voxelCoordinate(point.z)};
        const uint64_t index = localVoxel(voxel);
        const double distance = centerDistance(point, voxel);
        auto found = tile.points.find(index);
        if (found == tile.points.end()) {
          tile.points.emplace(index, Representative{point, distance});
          ++unique_points_;
          ++cache_points_;
          peak_cache_points_ = std::max(peak_cache_points_, cache_points_);
          tile.dirty = true;
        } else if (distance < found->second.squared_distance) {
          found->second = Representative{point, distance};
          tile.dirty = true;
        }
      }
    }
  }

  void flushTile(const TileKey& key, Tile& tile) {
    if (!tile.dirty) return;
    Cloud cloud;
    cloud.reserve(tile.points.size());
    for (const auto& entry : tile.points) cloud.push_back(entry.second.point);
    const std::string path = working_path_ + "/" + tileFilename(key);
    try {
      PcdStreamWriter<Point> output(path);
      output.append(cloud);
      output.finalize();
    } catch (const std::exception& error) {
      throw std::runtime_error("Cannot checkpoint map tile " + path + ": " + error.what());
    }
    tile.dirty = false;
  }

  void writeManifest() {
    const std::string temporary = working_path_ + "/manifest.txt.tmp";
    std::ofstream manifest;
    manifest.exceptions(std::ios::badbit | std::ios::failbit);
    manifest.open(temporary, std::ios::out | std::ios::trunc);
    manifest << "FAST_LIO_TILED_VOXEL_MAP 1\n" << std::setprecision(17)
             << "voxel_size " << options_.voxel_size << '\n'
             << "tile_size " << options_.tile_size << '\n'
             << "tile_voxels " << tile_voxels_ << '\n'
             << "unique_points " << unique_points_ << '\n'
             << "invalid_points " << invalid_points_ << '\n'
             << "format standard_binary_pcd_xyzi\n";
    manifest.close();
    if (std::rename(temporary.c_str(), (working_path_ + "/manifest.txt").c_str()) != 0)
      throw std::runtime_error("Cannot commit map manifest: " + working_path_);
  }

  void checkUsable() const {
    if (failure_) std::rethrow_exception(failure_);
    if (finalized_ || stopping_) throw std::logic_error("Cannot append to a finalized/stopped tiled map");
  }

  void barrier(Kind kind) {
    auto done = std::make_shared<std::promise<void>>();
    auto completion = done->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      checkUsable();
      Job job;
      job.kind = kind;
      job.done = done;
      jobs_.push_back(std::move(job));
    }
    ready_.notify_one();
    completion.get();
  }

  void publishStats() {
    // mutex_ must already be held. Called once per batch/barrier, not per point.
    stats_.unique_points = unique_points_;
    stats_.invalid_points = invalid_points_;
    stats_.active_tiles = tiles_.size();
    stats_.cache_points = cache_points_;
    stats_.peak_cache_points = peak_cache_points_;
  }

  void run() noexcept {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
        if (jobs_.empty()) return;
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      try {
        if (job.kind == Kind::Append) {
          process(*job.cloud);
        } else {
          for (auto& tile : tiles_) flushTile(tile.first, tile.second);
          writeManifest();
          if (job.kind == Kind::Finalize) exportCheckpoint(working_path_, final_path_);
        }
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (job.kind == Kind::Append) stats_.queued_points -= job.cloud->size();
          publishStats();
          if (job.kind == Kind::Finalize) finalized_ = stopping_ = true;
        }
        space_.notify_all();
        if (job.done) job.done->set_value();
        if (job.kind == Kind::Finalize) return;
      } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          failure_ = failure;
          stopping_ = true;
          stats_.failed = true;
          stats_.queued_points = 0;
          publishStats();
          for (auto& pending : jobs_)
            if (pending.done) pending.done->set_exception(failure);
          jobs_.clear();
        }
        if (job.done) job.done->set_exception(failure);
        space_.notify_all();
        ready_.notify_all();
        return;
      }
    }
  }

  std::string final_path_, working_path_;
  Options options_;
  int64_t tile_voxels_ = 1;
  std::unordered_map<TileKey, Tile, KeyHash> tiles_;
  std::list<TileKey> lru_;
  uint64_t unique_points_ = 0, invalid_points_ = 0;
  size_t cache_points_ = 0, peak_cache_points_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable ready_, space_;
  std::deque<Job> jobs_;
  Stats stats_;
  bool stopping_ = false, finalized_ = false;
  std::exception_ptr failure_;
  std::thread worker_;
};
