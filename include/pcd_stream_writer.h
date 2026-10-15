#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include <pcl/io/pcd_io.h>

// Keep the complete map on disk, with only a small packing buffer in RAM.
// The working PCD has a valid header after each completed append, so it can
// also be recovered if a run exits before finalization.
template <typename PointT>
class PcdStreamWriter {
 public:
  explicit PcdStreamWriter(const std::string& path)
      : final_path_(path), working_path_(path + ".in_progress." + std::to_string(getpid()) + ".pcd") {}

  void append(const pcl::PointCloud<PointT>& cloud) {
    if (finalized_) throw std::logic_error("Cannot append to a finalized PCD");
    if (cloud.empty()) return;
    if (count_ + cloud.size() > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("PCD point count exceeds UINT32_MAX");
    if (!stream_.is_open()) open(cloud);
    stream_.seekp(0, std::ios::end);
    const size_t block_points = std::max<size_t>(1, 65536 / point_bytes_);
    buffer_.resize(block_points * point_bytes_);
    for (size_t first = 0; first < cloud.size(); first += block_points) {
      const size_t end = std::min(cloud.size(), first + block_points);
      char* destination = buffer_.data();
      for (size_t index = first; index < end; ++index) {
        const char* source = reinterpret_cast<const char*>(&cloud[index]);
        for (const auto& field : fields_) {
          const size_t bytes = field.count * pcl::getFieldSize(field.datatype);
          std::memcpy(destination, source + field.offset, bytes);
          destination += bytes;
        }
      }
      stream_.write(buffer_.data(), (end - first) * point_bytes_);
    }
    // Commit payload before announcing the new count in the PCD header.
    stream_.flush();
    count_ += cloud.size();
    checkpoint();
  }

  void checkpoint() {
    if (!stream_.is_open()) return;
    writeCount(width_offset_);
    writeCount(points_offset_);
    stream_.flush();
    stream_.seekp(0, std::ios::end);
  }

  void finalize() {
    if (!stream_.is_open()) return;
    checkpoint();
    stream_.close();
    if (std::rename(working_path_.c_str(), final_path_.c_str()) != 0)
      throw std::runtime_error("Cannot rename completed PCD to " + final_path_);
    finalized_ = true;
  }

  uint64_t count() const { return count_; }
  size_t bufferBytes() const { return buffer_.capacity(); }
  const std::string& workingPath() const { return working_path_; }

 private:
  void open(const pcl::PointCloud<PointT>& cloud) {
    if (std::ifstream(working_path_).good())
      throw std::runtime_error("Existing map checkpoint must be preserved: " + working_path_);
    pcl::PCDWriter writer;
    std::string header = writer.generateHeader<PointT>(cloud, 0);
    replaceCount(header, "WIDTH ");
    replaceCount(header, "POINTS ");
    width_offset_ = header.find("\nWIDTH ") + 7;
    points_offset_ = header.find("\nPOINTS ") + 8;
    header += "DATA binary\n";
    for (const auto& field : pcl::getFields<PointT>()) {
      if (field.name == "_") continue;
      fields_.push_back(field);
      point_bytes_ += field.count * pcl::getFieldSize(field.datatype);
    }
    if (!point_bytes_) throw std::runtime_error("PCD point type has no fields");
    stream_.exceptions(std::ios::badbit | std::ios::failbit);
    stream_.open(working_path_, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
    stream_.write(header.data(), header.size());
    stream_.flush();
  }

  static void replaceCount(std::string& header, const std::string& name) {
    const size_t begin = header.find("\n" + name);
    if (begin == std::string::npos) throw std::runtime_error("PCD header missing " + name);
    const size_t value = begin + 1 + name.size();
    header.replace(value, header.find('\n', value) - value, std::string(19, ' ') + "0");
  }

  void writeCount(std::streamoff offset) {
    std::ostringstream text;
    text << std::setw(20) << count_;
    stream_.seekp(offset);
    stream_.write(text.str().data(), 20);
  }

  std::string final_path_, working_path_;
  std::fstream stream_;
  std::streamoff width_offset_ = 0, points_offset_ = 0;
  std::vector<pcl::PCLPointField> fields_;
  std::vector<char> buffer_;
  size_t point_bytes_ = 0;
  uint64_t count_ = 0;
  bool finalized_ = false;
};
