#!/usr/bin/env python3
"""Exercise lightweight map saving on a private ROS master with synthetic scans.

Use a FAST-LIO binary compiled with FAST_LIO_RUNTIME_DIR below /tmp, for example:
  python3 test_lightweight_mapping.py \
      --binary /tmp/fastlio_memory_fix/devel/lib/fast_lio/fastlio_mapping \
      --runtime-root /tmp/fastlio_memory_fix/runtime

No dataset, existing ROS master, or workspace PCD/Log file is accessed.
"""

import argparse
import json
import math
import mmap
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import xmlrpc.client

import numpy as np


def wait_for(check, description, timeout=20.0):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except (ConnectionError, OSError, xmlrpc.client.Error) as error:
            last_error = error
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for %s: %s" % (description, last_error))


def free_port():
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port != 11311:
            return port


def stop_owned_process(process):
    if process is None or process.poll() is not None:
        return
    # Signal only a child spawned here; never discover or stop existing nodes.
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def read_xyzi_pcd(path, exact_fields=True):
    header = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            assert line, "Missing PCD data header: %s" % path
            parts = line.decode("ascii").strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            header[parts[0]] = parts[1:]
            if parts[0] == "DATA":
                break
        fields = header["FIELDS"]
        if exact_fields:
            assert fields == ["x", "y", "z", "intensity"], header
        assert header["SIZE"] == ["4"] * len(fields), header
        assert header["TYPE"] == ["F"] * len(fields), header
        assert header["COUNT"] == ["1"] * len(fields), header
        assert header["DATA"] == ["binary"], header
        count = int(header["POINTS"][0])
        assert int(header["WIDTH"][0]) == count and header["HEIGHT"] == ["1"], header
        payload = stream.read()
    assert len(payload) == count * 4 * len(fields), (len(payload), count)
    data = np.frombuffer(payload, dtype="<f4").reshape(count, len(fields))
    return data[:, [fields.index(name) for name in ("x", "y", "z", "intensity")]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="/tmp/fastlio_memory_fix/devel/lib/fast_lio/fastlio_mapping")
    parser.add_argument("--runtime-root", default="/tmp/fastlio_memory_fix/runtime")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--save-mode", choices=("lightweight", "streaming", "interval"), default="lightweight")
    parser.add_argument("--moving", action="store_true", help="Translate synthetic scans to produce multiple keyframes")
    args = parser.parse_args()
    binary = Path(args.binary).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    if not binary.is_file():
        parser.error("FAST-LIO binary does not exist: %s" % binary)
    if runtime_root == Path("/tmp") or Path("/tmp") not in runtime_root.parents:
        parser.error("The binary's compiled runtime root must be a directory below /tmp")
    if args.frames < 20:
        parser.error("Use at least 20 frames to exercise static keyframe rejection")
    # ROOT_DIR is compiled into the binary. Checking only a CLI runtime path would
    # still allow the normal workspace binary to truncate the user's debug logs.
    with binary.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if data.find((str(runtime_root) + "/").encode()) < 0:
            parser.error("Binary does not contain the requested /tmp ROOT_DIR; rebuild it with "
                         "-DFAST_LIO_RUNTIME_DIR=" + str(runtime_root))
    for name in ("Log", "PCD"):
        (runtime_root / name).mkdir(parents=True, exist_ok=True)

    test_dir = Path(tempfile.mkdtemp(prefix="fastlio_lightweight_test_", dir="/tmp"))
    map_path = test_dir / "map.pcd"
    port = free_port()
    os.environ.update(ROS_MASTER_URI="http://127.0.0.1:%d" % port,
                      ROS_IP="127.0.0.1", ROS_HOME=str(test_dir / "ros_home"),
                      ROS_LOG_DIR=str(test_dir / "ros_logs"), OPENBLAS_NUM_THREADS="1",
                      OMP_NUM_THREADS="3")
    os.environ.pop("ROS_HOSTNAME", None)
    os.environ.pop("ROS_NAMESPACE", None)

    import rospy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, PointCloud2, PointField
    from sensor_msgs import point_cloud2
    from std_msgs.msg import Header
    from std_srvs.srv import Trigger

    master_process = mapping_process = None
    master_log = (test_dir / "master.log").open("w")
    mapping_log = (test_dir / "mapping.log").open("w")
    try:
        master_process = subprocess.Popen(
            ["/opt/ros/noetic/bin/roscore", "-p", str(port)],
            stdout=master_log, stderr=subprocess.STDOUT)
        master = xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"])
        wait_for(lambda: master.getPid("/fastlio_lightweight_test")[0] == 1, "private ROS master")
        rospy.init_node("fastlio_lightweight_test", anonymous=True, disable_signals=True)
        params = {
            # No /clock is needed: processing and saving must also drain at bag end.
            "/use_sim_time": True,
            "/buffers/max_lidar_frames": 4,
            "/buffers/max_imu_messages": 1000,
            "/buffers/lidar_sub_queue_size": 100,
            "/buffers/imu_sub_queue_size": 1000,
            "/common/lid_topic": "/fastlio_input/points",
            "/common/imu_topic": "/fastlio_input/imu",
            "/common/time_sync_en": False,
            "/common/time_offset_lidar_to_imu": 0.0,
            "/preprocess/lidar_type": 2,
            "/preprocess/timestamp_unit": 0,
            "/preprocess/scan_line": 16,
            "/preprocess/scan_rate": 10,
            "/preprocess/blind": 0.01,
            "/point_filter_num": 1,
            "/feature_extract_enable": False,
            "/filter_size_surf": 0.3,
            "/filter_size_map": 0.4,
            "/cube_side_length": 400.0,
            "/mapping/det_range": 100.0,
            "/mapping/max_point_range": 10.0,
            "/max_iteration": 3,
            "/publish/path_en": False,
            "/publish/scan_publish_en": False,
            "/publish/dense_publish_en": False,
            "/publish/scan_bodyframe_pub_en": False,
            "/publish/map_publish_en": False,
            "/runtime_pos_log_enable": False,
            "/mapping/extrinsic_est_en": False,
            "/mapping/extrinsic_T": [0.15, -0.25, 0.35],
            "/mapping/extrinsic_R": [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "/pcd_save/pcd_save_en": True,
            "/pcd_save/lightweight_en": args.save_mode == "lightweight",
            "/pcd_save/interval": 7 if args.save_mode == "interval" else -1,
            "/pcd_save/voxel_size": 0.25,
            "/pcd_save/keyframe_translation": 0.75,
            "/pcd_save/keyframe_rotation_deg": 10.0,
            "/pcd_save/keyframe_max_interval": 2.0,
            "/pcd_save/tile_size": 4.0,
            "/pcd_save/max_active_tiles": 2,
            "/pcd_save/max_queued_points": 1024,
            "/map_file_path": str(map_path),
        }
        for name, value in params.items():
            rospy.set_param(name, value)
        odometry = {}

        def receive_odometry(message):
            pose = message.pose.pose
            odometry[round(message.header.stamp.to_sec(), 6)] = (
                np.array([pose.position.x, pose.position.y, pose.position.z]),
                np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]))

        odometry_subscriber = rospy.Subscriber("/Odometry", Odometry, receive_odometry, queue_size=100)
        mapping_process = subprocess.Popen(
            [str(binary), "__name:=fastlio_mapping", "__ns:=/"],
            stdout=mapping_log, stderr=subprocess.STDOUT)
        rospy.wait_for_service("/fastlio_mapping/get_processing_state", timeout=20)
        state_service = rospy.ServiceProxy("/fastlio_mapping/get_processing_state", Trigger)
        checkpoint_service = rospy.ServiceProxy("/fastlio_mapping/save_map", Trigger)

        def state():
            assert mapping_process.poll() is None, "FAST-LIO exited; inspect %s" % (test_dir / "mapping.log")
            response = state_service()
            assert response.success, response.message
            return json.loads(response.message)

        lidar_publisher = rospy.Publisher("/fastlio_input/points", PointCloud2, queue_size=100)
        imu_publisher = rospy.Publisher("/fastlio_input/imu", Imu, queue_size=1000)
        wait_for(lambda: lidar_publisher.get_num_connections(), "lidar subscription")
        wait_for(lambda: imu_publisher.get_num_connections(), "IMU subscription")
        fields = [PointField("x", 0, PointField.FLOAT32, 1),
                  PointField("y", 4, PointField.FLOAT32, 1),
                  PointField("z", 8, PointField.FLOAT32, 1),
                  PointField("intensity", 12, PointField.FLOAT32, 1),
                  PointField("time", 16, PointField.FLOAT32, 1),
                  PointField("ring", 20, PointField.UINT16, 1)]
        geometry = []
        for i in range(11):
            for j in range(11):
                a, b = -1.96 + i * 0.4, -1.46 + j * 0.3
                geometry.extend([(4.125, a, b, 10.0), (a, 4.125, b, 20.0),
                                 (a, b, -2.125, 30.0)])
        geometry.extend([(12.0, 0.0, 0.0, 9999.0), (0.0, 16.0, 0.0, 9999.0)])
        imu_index = -1
        last_scan_end = 0.0
        for index in range(args.frames):
            scan_stamp = 10.0 + index * 0.1
            before = state()["imu_buffer"]
            new_samples = 0
            while 10.0 + imu_index * 0.005 <= scan_stamp + 0.105 + 1e-8:
                message = Imu()
                message.header.stamp = rospy.Time.from_sec(10.0 + imu_index * 0.005)
                message.linear_acceleration.z = 9.81
                imu_publisher.publish(message)
                imu_index += 1
                new_samples += 1
            # With no pending lidar, all IMU samples stay buffered. Acknowledge
            # them before the scan so asynchronous transports cannot drop scans.
            wait_for(lambda: state()["imu_buffer"] == before + new_samples, "IMU samples for scan %d" % index)
            header = Header(stamp=rospy.Time.from_sec(scan_stamp), frame_id="lidar")
            displacement = index * 0.08 if args.moving else 0.0
            points = [(x - displacement if intensity != 9999.0 else x, y, z, intensity,
                       i * 0.099 / (len(geometry) - 1), i % 16)
                      for i, (x, y, z, intensity) in enumerate(geometry)]
            lidar_publisher.publish(point_cloud2.create_cloud(header, fields, points))
            wait_for(lambda: state()["received_frames"] == index + 1 and
                     state()["lidar_buffer"] == 0, "processed or initialization scan %d" % index)
            last_scan_end = scan_stamp + 0.099

        observed = state()
        assert observed["processed_frames"] >= args.frames - 5, observed
        assert abs(observed["last_processed_stamp"] - last_scan_end) < 1e-5, observed
        assert observed["dropped_lidar_frames"] == 0 and observed["dropped_imu_messages"] == 0, observed
        if args.moving:
            assert observed["save_keyframes"] > 1, observed
        else:
            assert observed["save_keyframes"] == 1, observed
        assert observed["save_skipped_frames"] == observed["processed_frames"] - observed["save_keyframes"], observed
        assert 5 < observed["local_map_points"], observed
        assert math.isfinite(observed["last_mapping_time_ms"]) and observed["last_mapping_time_ms"] >= 0, observed
        assert observed["save_queued_points"] <= params["/pcd_save/max_queued_points"], observed
        assert observed["save_cache_points"] <= 2 * (4.0 / 0.25) ** 3, observed

        checkpoint = checkpoint_service()
        if args.save_mode != "interval":
            assert checkpoint.success, checkpoint.message
            checkpoint_path = Path(checkpoint.message).resolve()
            assert test_dir in checkpoint_path.parents and checkpoint_path.exists(), checkpoint.message
        else:
            assert not checkpoint.success, checkpoint.message  # Legacy chunks have no checkpoint service.
        observed = state()
        if args.save_mode == "lightweight":
            assert 0 < observed["saved_points"] <= (len(geometry) - 2) * observed["save_keyframes"], observed
        assert observed["save_queued_points"] == 0, observed
        trajectory_checkpoint = test_dir / ("trajectory.pcd.in_progress.%d.pcd" % mapping_process.pid)
        poses_checkpoint = test_dir / ("keyframe_poses.tum.in_progress.%d" % mapping_process.pid)
        checkpoint_trajectory = read_xyzi_pcd(trajectory_checkpoint)
        checkpoint_poses = np.loadtxt(poses_checkpoint, ndmin=2)
        assert len(checkpoint_trajectory) == len(checkpoint_poses) == observed["save_keyframes"], observed
        stop_owned_process(mapping_process)
        assert mapping_process.returncode == 0, "FAST-LIO did not finalize cleanly; inspect %s" % mapping_log.name
        map_files = sorted(test_dir.glob("map_*.pcd"))
        if map_path.is_file():
            final_map = read_xyzi_pcd(map_path, exact_fields=args.save_mode == "lightweight")
            assert len(final_map) == observed["saved_points"], (len(final_map), observed)
            map_files.append(map_path)
        assert map_files, "No map was exported: %s" % test_dir
        saved = np.concatenate([read_xyzi_pcd(path, exact_fields=args.save_mode == "lightweight") for path in map_files])
        assert np.isfinite(saved).all(), "Non-finite saved point"
        assert np.max(np.linalg.norm(saved[:, :3], axis=1)) <= 10.6, "Range crop failed"
        assert not np.any(saved[:, 3] == 9999.0), "Out-of-range points reached the saved map"
        if args.save_mode == "lightweight":
            voxels = np.floor(saved[:, :3].astype(np.float64) / 0.25).astype(np.int64)
            assert len(np.unique(voxels, axis=0)) == len(saved), "Global saved voxels are not unique"
        trajectory = read_xyzi_pcd(test_dir / "trajectory.pcd")
        poses = np.loadtxt(test_dir / "keyframe_poses.tum", ndmin=2)
        assert len(trajectory) == len(poses) == observed["save_keyframes"], observed
        np.testing.assert_array_equal(trajectory, checkpoint_trajectory)
        np.testing.assert_array_equal(poses, checkpoint_poses)
        np.testing.assert_allclose(trajectory[:, :3], poses[:, 1:4], atol=1e-6)
        np.testing.assert_array_equal(trajectory[:, 3], np.arange(len(poses)))
        np.testing.assert_allclose(np.linalg.norm(poses[:, 4:], axis=1), 1.0, atol=1e-10)
        assert np.all(np.diff(poses[:, 0]) > 0), "Keyframe acquisition times are not increasing"
        for row in poses:
            stamp = round(row[0], 6)
            wait_for(lambda: stamp in odometry, "odometry for keyframe %s" % stamp)
            position, quaternion = odometry[stamp]
            extrinsic = np.array(params["/mapping/extrinsic_T"])
            vector, scalar = quaternion[:3], quaternion[3]
            cross = 2.0 * np.cross(vector, extrinsic)
            expected_position = position + extrinsic + scalar * cross + np.cross(vector, cross)
            np.testing.assert_allclose(row[1:4], expected_position, atol=1e-8)
            extrinsic_q = np.array([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
            expected_q = np.r_[scalar * extrinsic_q[:3] + extrinsic_q[3] * vector +
                               np.cross(vector, extrinsic_q[:3]),
                               scalar * extrinsic_q[3] - np.dot(vector, extrinsic_q[:3])]
            np.testing.assert_allclose(abs(np.dot(row[4:], expected_q)), 1.0, atol=1e-8)
        assert not trajectory_checkpoint.exists() and not poses_checkpoint.exists()
        print(json.dumps({"result": "PASS", "master_port": port, "logs": str(test_dir),
                          "save_mode": args.save_mode, "moving": args.moving,
                          "keyframe_poses": len(poses),
                          "final_map": str(map_path), "final_map_points": len(saved),
                          "state": observed}, indent=2, sort_keys=True))
    finally:
        stop_owned_process(mapping_process)
        if rospy.core.is_initialized():
            rospy.signal_shutdown("isolated lightweight test complete")
        stop_owned_process(master_process)
        mapping_log.close()
        master_log.close()


if __name__ == "__main__":
    main()
