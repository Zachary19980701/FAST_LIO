#!/usr/bin/env python3
"""Verify FAST-LIO buffer bounds on an isolated ROS master.

Run with the ROS environment sourced and a binary compiled with ROOT_DIR in /tmp:
    python3 test_memory_buffers.py --binary /tmp/fastlio_memory_fix/devel/lib/fast_lio/fastlio_mapping

The existing ROS master and running FAST-LIO processes are never accessed.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import xmlrpc.client


def wait_for(check, description, timeout=15.0):
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
    # Only the directly spawned process is signalled, never a process name.
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="/tmp/fastlio_memory_fix/devel/lib/fast_lio/fastlio_mapping")
    parser.add_argument("--runtime-root", default="/tmp/fastlio_memory_fix/runtime")
    args = parser.parse_args()
    binary = Path(args.binary).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    if not binary.is_file():
        parser.error("FAST-LIO binary does not exist: %s" % binary)
    if os.path.commonpath([str(runtime_root), "/tmp"]) != "/tmp":
        parser.error("The test binary's runtime root must be under /tmp")
    for directory in ("Log", "PCD"):
        (runtime_root / directory).mkdir(parents=True, exist_ok=True)

    test_dir = Path(tempfile.mkdtemp(prefix="fastlio_buffer_test_", dir="/tmp"))
    port = free_port()
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:%d" % port
    os.environ["ROS_IP"] = "127.0.0.1"
    os.environ.pop("ROS_HOSTNAME", None)
    os.environ.pop("ROS_NAMESPACE", None)
    os.environ["ROS_HOME"] = str(test_dir / "ros_home")
    os.environ["ROS_LOG_DIR"] = str(test_dir / "ros_logs")

    import rospy
    from sensor_msgs.msg import Imu, PointCloud2, PointField
    from sensor_msgs import point_cloud2
    from std_msgs.msg import Header
    from std_srvs.srv import Trigger

    master_process = None
    mapping_process = None
    master_log = (test_dir / "master.log").open("w")
    mapping_log = (test_dir / "mapping.log").open("w")
    try:
        master_process = subprocess.Popen(
            ["/opt/ros/noetic/bin/roscore", "-p", str(port)],
            stdout=master_log, stderr=subprocess.STDOUT,
        )
        master = xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"])
        wait_for(lambda: master.getPid("/fastlio_buffer_test")[0] == 1, "isolated ROS master")
        rospy.init_node("fastlio_buffer_test", anonymous=True, disable_signals=True)
        params = {
            # No /clock is published: callbacks must keep running after bag playback ends.
            "/use_sim_time": True,
            "/buffers/max_lidar_frames": 3,
            "/buffers/max_imu_messages": 5,
            "/buffers/lidar_sub_queue_size": 100,
            "/buffers/imu_sub_queue_size": 100,
            "/common/lid_topic": "/fastlio_input/points",
            "/common/imu_topic": "/fastlio_input/imu",
            "/common/time_sync_en": False,
            "/common/time_offset_lidar_to_imu": 0.0,
            "/preprocess/lidar_type": 2,
            "/preprocess/timestamp_unit": 0,
            "/preprocess/scan_line": 16,
            "/preprocess/blind": 0.01,
            "/point_filter_num": 1,
            "/feature_extract_enable": False,
            "/publish/path_en": False,
            "/publish/scan_publish_en": False,
            "/publish/dense_publish_en": False,
            "/publish/scan_bodyframe_pub_en": False,
            "/publish/map_publish_en": False,
            "/runtime_pos_log_enable": False,
            "/pcd_save/pcd_save_en": False,
            "/mapping/extrinsic_est_en": False,
            "/mapping/extrinsic_T": [0.0, 0.0, 0.0],
            "/mapping/extrinsic_R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
        for name, value in params.items():
            rospy.set_param(name, value)
        mapping_process = subprocess.Popen(
            [str(binary), "__name:=fastlio_mapping", "__ns:=/"],
            stdout=mapping_log, stderr=subprocess.STDOUT,
        )
        rospy.wait_for_service("/fastlio_mapping/get_processing_state", timeout=15)
        state_service = rospy.ServiceProxy("/fastlio_mapping/get_processing_state", Trigger)

        def state():
            if mapping_process.poll() is not None:
                raise AssertionError("FAST-LIO exited; inspect %s" % (test_dir / "mapping.log"))
            response = state_service()
            assert response.success, response.message
            return json.loads(response.message)

        lidar_publisher = rospy.Publisher("/fastlio_input/points", PointCloud2, queue_size=100)
        imu_publisher = rospy.Publisher("/fastlio_input/imu", Imu, queue_size=100)
        wait_for(lambda: lidar_publisher.get_num_connections() > 0, "lidar subscription")
        wait_for(lambda: imu_publisher.get_num_connections() > 0, "IMU subscription")
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("intensity", 12, PointField.FLOAT32, 1),
            PointField("time", 16, PointField.FLOAT32, 1),
            PointField("ring", 20, PointField.UINT16, 1),
        ]

        def publish_lidar(stamp):
            header = Header(stamp=rospy.Time.from_sec(stamp), frame_id="lidar")
            points = [(5.0 + i * 0.02, i * 0.01, 0.1, 10.0, i * 0.099 / 63, i % 16)
                      for i in range(64)]
            lidar_publisher.publish(point_cloud2.create_cloud(header, fields, points))

        # Acknowledge every frame so ROS transport drops cannot affect the result.
        for index in range(12):
            publish_lidar(10.0 + index * 0.1)
            wait_for(lambda: state()["received_frames"] == index + 1, "lidar frame %d" % index)
        lidar_state = state()
        assert lidar_state["lidar_buffer"] == 3, lidar_state
        assert lidar_state["lidar_time_buffer"] == 3, lidar_state
        assert lidar_state["dropped_lidar_frames"] == 9, lidar_state
        assert lidar_state["processed_frames"] == 0, lidar_state
        assert abs(lidar_state["oldest_pending_stamp"] - 10.9) < 1e-8, lidar_state

        # Deliberately old IMU stamps cannot release any of the lidar frames.
        for index in range(8):
            message = Imu()
            message.header.stamp = rospy.Time.from_sec(index * 0.01)
            message.linear_acceleration.z = 9.81
            imu_publisher.publish(message)
            expected_size = min(index + 1, 5)
            expected_dropped = max(index + 1 - 5, 0)
            wait_for(lambda: state()["imu_buffer"] == expected_size and
                     state()["dropped_imu_messages"] == expected_dropped, "IMU sample %d" % index)
        imu_state = state()
        assert imu_state["imu_buffer"] == 5, imu_state
        assert imu_state["dropped_imu_messages"] == 3, imu_state
        assert imu_state["processed_frames"] == 0, imu_state

        publish_lidar(9.0)
        wait_for(lambda: state()["received_frames"] == 13, "lidar timestamp rollback")
        rollback_state = state()
        assert rollback_state["lidar_buffer"] == 1, rollback_state
        assert rollback_state["lidar_time_buffer"] == 1, rollback_state
        assert rollback_state["oldest_pending_stamp"] == 9.0, rollback_state
        assert rollback_state["imu_buffer"] == 5, rollback_state
        assert rollback_state["dropped_lidar_frames"] == 9, rollback_state
        assert rollback_state["processed_frames"] == 0, rollback_state

        # Release the first scan with the existing early samples. The new late
        # sample remains buffered, so the next scan has no IMU in its interval.
        late_imu = Imu()
        late_imu.header.stamp = rospy.Time.from_sec(10.0)
        late_imu.linear_acceleration.z = 9.81
        imu_publisher.publish(late_imu)
        wait_for(lambda: state()["lidar_buffer"] == 0 and state()["imu_buffer"] == 1,
                 "first scan released by late IMU")
        publish_lidar(9.1)
        wait_for(lambda: state()["received_frames"] == 14 and state()["lidar_buffer"] == 0,
                 "scan with missing IMU interval skipped")
        gap_state = state()
        assert gap_state["lidar_time_buffer"] == 0, gap_state
        assert gap_state["dropped_lidar_frames"] == 10, gap_state
        assert gap_state["imu_buffer"] == 1, gap_state
        assert gap_state["processed_frames"] == 0, gap_state

        print(json.dumps({"result": "PASS", "master_port": port, "logs": str(test_dir),
                          "lidar_limit": lidar_state, "imu_limit": imu_state,
                          "timestamp_rollback": rollback_state,
                          "missing_imu_interval": gap_state}, indent=2, sort_keys=True))
    finally:
        stop_owned_process(mapping_process)
        if rospy.core.is_initialized():
            rospy.signal_shutdown("isolated buffer test complete")
        stop_owned_process(master_process)
        mapping_log.close()
        master_log.close()


if __name__ == "__main__":
    main()
