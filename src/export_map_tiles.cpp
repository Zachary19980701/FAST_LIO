#include <iostream>
#include "tiled_voxel_map_writer.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: fastlio_export_tiles CHECKPOINT_DIRECTORY OUTPUT.pcd\n"
              << "Stop the original writer before exporting its checkpoints.\n";
    return 2;
  }
  try {
    const auto points = TiledVoxelMapWriter::exportCheckpoint(argv[1], argv[2]);
    std::cout << "Exported " << points << " XYZI points to " << argv[2] << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Tile export failed: " << error.what() << '\n';
    return 1;
  }
}
