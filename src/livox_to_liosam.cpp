#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/Imu.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

// 目标点型：XYZI + ring(uint16) + time(float) —— LIO-SAM常用
struct PointXYZIRT {
  PCL_ADD_POINT4D;
  float intensity;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
  (float, x, x)(float, y, y)(float, z, z)
  (float, intensity, intensity)
  (uint16_t, ring, ring)
  (float, time, time)
)

class PC2RingTimeNode {
public:
  PC2RingTimeNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
    // --- 点云参数 ---
    pnh.param<std::string>("in_cloud", in_cloud_, "/lidar");
    pnh.param<std::string>("out_cloud", out_cloud_, "/points_raw");
    pnh.param<int>("N_SCAN", N_SCAN_, 32);
    pnh.param<double>("ang_min_deg", ang_min_deg_, -25);
    pnh.param<double>("ang_max_deg", ang_max_deg_,  15);
    pnh.param<bool>("drop_out_of_fov", drop_out_of_fov_, true);
    pnh.param<double>("scan_period", scan_period_, 0.1);
    pnh.param<std::string>("time_field_name", time_field_name_, "time");

    // --- IMU 参数 ---
    pnh.param<std::string>("in_imu", in_imu_, "/imu");
    pnh.param<std::string>("out_imu", out_imu_, "/imu_correct");
    pnh.param<bool>("rotate_imu_to_lidar", rotate_imu_to_lidar_, false); // 是否把IMU旋到LiDAR帧
    pnh.param<std::string>("out_imu_frame", out_imu_frame_, "lidar_link"); // 输出IMU的frame_id
    pnh.param<bool>("acc_in_g", acc_in_g_, true);     // 输入线加速度是否以“g”为单位
    pnh.param<bool>("gyro_in_dps", gyro_in_dps_, false); // 输入角速度是否为度每秒
    // T_lb（lidar->imu）行优先 3x3
    std::vector<double> rot_lb_vec;
    rot_lb_vec.reserve(9);
    if (!pnh.getParam("extrinsicRot_lb", rot_lb_vec) || rot_lb_vec.size() != 9) {
      // 默认单位阵（若你倒装请在参数里显式给出 diag(1,-1,-1)）
      rot_lb_vec = {1,0,0, 0,1,0, 0,0,1};
    }
    R_lb_ << rot_lb_vec[0], rot_lb_vec[1], rot_lb_vec[2],
             rot_lb_vec[3], rot_lb_vec[4], rot_lb_vec[5],
             rot_lb_vec[6], rot_lb_vec[7], rot_lb_vec[8];
    // 我们需要的是 IMU->LiDAR（R_bl），用于把IMU量旋到LiDAR帧：R_bl = R_lb^T
    R_bl_ = R_lb_.transpose();
    q_bl_ = Eigen::Quaterniond(R_bl_); // 旋转四元数

    sub_cloud_ = nh.subscribe(in_cloud_, 5, &PC2RingTimeNode::cloudCb, this);
    pub_cloud_ = nh.advertise<sensor_msgs::PointCloud2>(out_cloud_, 5);

    sub_imu_ = nh.subscribe(in_imu_, 100, &PC2RingTimeNode::imuCb, this);
    pub_imu_ = nh.advertise<sensor_msgs::Imu>(out_imu_, 100);

    ROS_INFO_STREAM("pc2_ring_time: cloud sub=" << in_cloud_ << " pub=" << out_cloud_
                    << " N_SCAN=" << N_SCAN_ << " ang=[" << ang_min_deg_ << "," << ang_max_deg_ << "]");
    ROS_INFO_STREAM("imu_convert:  imu sub=" << in_imu_ << " pub=" << out_imu_
                    << " rotate_to_lidar=" << (rotate_imu_to_lidar_ ? "true":"false")
                    << " acc_in_g=" << (acc_in_g_ ? "true":"false")
                    << " gyro_in_dps=" << (gyro_in_dps_ ? "true":"false"));
    ROS_INFO_STREAM("R_lb (lidar->imu):\n" << R_lb_ << "\nR_bl (imu->lidar):\n" << R_bl_);
  }

private:
  // ---------------- 点云处理 ----------------
  static bool hasField(const sensor_msgs::PointCloud2ConstPtr& msg, const std::string& name) {
    for (const auto& f : msg->fields) if (f.name == name) return true;
    return false;
  }

  void cloudCb(const sensor_msgs::PointCloud2ConstPtr& msg) {
    pcl::PointCloud<pcl::PointXYZI> in;
    pcl::fromROSMsg(*msg, in);

    const bool has_time_in = hasField(msg, time_field_name_);
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_time;
    if (has_time_in) it_time.reset(new sensor_msgs::PointCloud2ConstIterator<float>(*msg, time_field_name_));

    const double ang_min = ang_min_deg_ * M_PI / 180.0;
    const double ang_max = ang_max_deg_ * M_PI / 180.0;
    const double ang_span = ang_max - ang_min;

    pcl::PointCloud<PointXYZIRT> out;
    out.reserve(in.size());

    size_t out_of_fov = 0;
    const size_t n = in.size();

    for (size_t i = 0; i < n; ++i) {
      const auto& p = in.points[i];
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;

      const double r_xy = std::hypot(p.x, p.y);
      const double elev = std::atan2((double)p.z, r_xy);

      int ring = -1;
      if (ang_span > 1e-6) {
        const double t = (elev - ang_min) / ang_span;
        const double ridx = t * (N_SCAN_ - 1);
        if (t < 0.0 || t > 1.0) {
          if (drop_out_of_fov_) { out_of_fov++; continue; }
          ring = std::min(std::max((int)std::lround(ridx), 0), N_SCAN_ - 1);
        } else {
          ring = (int)std::lround(ridx);
        }
      } else {
        ring = 0;
      }

      PointXYZIRT q;
      q.x = p.x; q.y = p.y; q.z = p.z;
      q.intensity = p.intensity;
      q.ring = static_cast<uint16_t>(ring);

      if (has_time_in) {
        q.time = **it_time;
        ++(*it_time);
      } else {
        q.time = (n > 1) ? (float)(scan_period_ * (double)i / (double)(n - 1)) : 0.0f;
      }

      out.push_back(q);
    }

    out.width  = static_cast<uint32_t>(out.size());
    out.height = 1;

    // debug: 打印点云数量、time 范围、header 时间
    {
      float tmin=1e9f, tmax=-1e9f;
      for (const auto &pt : out) { tmin = std::min(tmin, pt.time); tmax = std::max(tmax, pt.time); }
      ROS_INFO("pc2_ring_time: publish %zu points header.stamp=%.3f time[min, max]=[%.6f, %.6f]",
               out.size(), msg->header.stamp.toSec(), (double)tmin, (double)tmax);
    }

    sensor_msgs::PointCloud2 out_msg;
    pcl::toROSMsg(out, out_msg);
    out_msg.header = msg->header; // 保留原始时间戳/坐标系
    pub_cloud_.publish(out_msg);

    if (out_of_fov > 0) {
      ROS_DEBUG("pc2_ring_time: dropped %zu out-of-FOV points (%.2f%%)",
                out_of_fov, 100.0 * out_of_fov / std::max<size_t>(1, in.size()));
    }
  }

  // ---------------- IMU处理 ----------------
  static void rotateCov(const Eigen::Matrix3d& R, const boost::array<double,9>& in, boost::array<double,9>& out) {
    Eigen::Matrix3d S;
    S << in[0], in[1], in[2],
         in[3], in[4], in[5],
         in[6], in[7], in[8];
    if (!std::isfinite(S(0,0)) && !std::isfinite(S(1,1)) && !std::isfinite(S(2,2))) {
      // 常见的是全0或-1表示未知；直接拷贝
      out = in;
      return;
    }
    Eigen::Matrix3d So = R * S * R.transpose();
    out[0]=So(0,0); out[1]=So(0,1); out[2]=So(0,2);
    out[3]=So(1,0); out[4]=So(1,1); out[5]=So(1,2);
    out[6]=So(2,0); out[7]=So(2,1); out[8]=So(2,2);
  }

  void imuCb(const sensor_msgs::ImuConstPtr& msg) {
    sensor_msgs::Imu out = *msg; // 先拷贝一份，逐项修改
    // 1) 统一单位
    Eigen::Vector3d acc(msg->linear_acceleration.x,
                        msg->linear_acceleration.y,
                        msg->linear_acceleration.z);
    Eigen::Vector3d gyr(msg->angular_velocity.x,
                        msg->angular_velocity.y,
                        msg->angular_velocity.z);
    if (acc_in_g_)  acc *= 9.81;             // g -> m/s^2
    if (gyro_in_dps_) gyr *= (M_PI / 180.0);  // deg/s -> rad/s

    // 2) 旋到LiDAR帧（若需要）
    if (rotate_imu_to_lidar_) {
      acc = R_bl_ * acc;
      gyr = R_bl_ * gyr;

      // 旋转协方差（若非零/非默认）
      rotateCov(R_bl_, msg->linear_acceleration_covariance, out.linear_acceleration_covariance);
      rotateCov(R_bl_, msg->angular_velocity_covariance,   out.angular_velocity_covariance);

      // 姿态也可旋转（虽然LIO-SAM不用orientation，但我们仍保持一致性）
      const Eigen::Quaterniond q_in(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
      const Eigen::Quaterniond q_out = q_bl_ * q_in; // 先IMU自身，再到LiDAR
      out.orientation.w = q_out.w();
      out.orientation.x = q_out.x();
      out.orientation.y = q_out.y();
      out.orientation.z = q_out.z();

      out.header.frame_id = out_imu_frame_; // 通常设为与点云一致的LiDAR坐标系
    }

    // 3) 写回数据
    out.linear_acceleration.x = acc.x();
    out.linear_acceleration.y = acc.y();
    out.linear_acceleration.z = acc.z();
    out.angular_velocity.x = gyr.x();
    out.angular_velocity.y = gyr.y();
    out.angular_velocity.z = gyr.z();

    pub_imu_.publish(out);
  }

  // ROS
  ros::Subscriber sub_cloud_;
  ros::Publisher  pub_cloud_;
  ros::Subscriber sub_imu_;
  ros::Publisher  pub_imu_;

  // 点云参数
  std::string in_cloud_, out_cloud_, time_field_name_;
  int N_SCAN_;
  double ang_min_deg_, ang_max_deg_, scan_period_;
  bool drop_out_of_fov_;

  // IMU参数
  std::string in_imu_, out_imu_, out_imu_frame_;
  bool rotate_imu_to_lidar_, acc_in_g_, gyro_in_dps_;
  Eigen::Matrix3d R_lb_;     // lidar->imu
  Eigen::Matrix3d R_bl_;     // imu->lidar = R_lb^T
  Eigen::Quaterniond q_bl_;  // imu->lidar
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "pc2_ring_time_livox");
  ros::NodeHandle nh, pnh("~");
  PC2RingTimeNode node(nh, pnh);
  ros::spin();
  return 0;
}