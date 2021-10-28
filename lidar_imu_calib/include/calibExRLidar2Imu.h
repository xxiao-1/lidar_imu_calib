/*
 * brief: calibrate extrincs rotation between multi-layer lidar and imu
 * author: chennuo0125@163.com
*/

#pragma once

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include "../../ndt_omp/include/pclomp/ndt_omp.h"

using namespace std;
using PointT = pcl::PointXYZI;
using CloudT = pcl::PointCloud<PointT>;
typedef std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
    EigenAffineVector;

struct LidarData
{
    double stamp;
    CloudT::Ptr cloud;
};
// 手眼标定，AB都为delta矩阵，所以imu chassis 的rotation和translation都是delta值
struct LidarFrame
{
    double stamp;
    Eigen::Matrix4d T;
    Eigen::Matrix4d gT;
    CloudT::Ptr cloud{nullptr};

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct SensorFrame
{
    double stamp;
    Eigen::Quaterniond rot;
    Eigen::Vector3d tra;
};

struct Frame
{
    Eigen::Quaterniond rot;
    Eigen::Vector3d tra;
};

struct ImuData
{
    double stamp;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
    Eigen::Quaterniond rot;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ChassisData
{
    double stamp;
    Eigen::Vector3d velocity;
    Eigen::Vector3d angVelocity;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class CalibExRLidarImu
{
public:
    CalibExRLidarImu();
    ~CalibExRLidarImu();

    //@brief: set init extrinsic if have
    void setInitExR(Eigen::Vector3d init_R);

    //@brief: add lidar data and calculate lidar odometry
    void addLidarData(const LidarData &data);

    //@brief: add imu data and cache
    void addImuFrame(const SensorFrame &data);

    //@brief: add chassis data and cache
    void addChassisFrame(const SensorFrame &data);

    //@brief: integration imu data, align lidar odom and imu
    void calibLidar2Imu();

    //@brief:  align lidar odom and chassis
    void calibLidar2Chassis();

    //@brief: align chassis odom and imu
    void calibMulti();

private:
    //@brief: interpolated attitude from start attitude to end attitude by scale
    Eigen::Quaterniond getInterpolatedAttitude(const Eigen::Quaterniond &q_s_w, const Eigen::Quaterniond &q_e_w, double scale);

    Eigen::Vector3d getInterpolatedTranslation(const Eigen::Vector3d &t_s_w, const Eigen::Vector3d &t_e_w, double scale);

    Eigen::Matrix3d skew(Eigen::Vector3d u);
    //@brief: update relative transform between neighbor lidar frame by aligned imu data
    void optimize(string sensorName, vector<pair<LidarFrame, SensorFrame>> &aligned_sensor_buffer_);

    //@brief: solve least square answer by constraints
    Frame solve(const vector<pair<Frame, Frame>> &corres);

    Eigen::Matrix4d solveX(const vector<pair<Frame, Frame>> &corres);

    Eigen::Vector3d init_R_{0.0, 0.0, 0.0};
    CloudT::Ptr last_lidar_cloud_{nullptr};
    vector<LidarFrame> lidar_buffer_;                                    // record relative transform between neighbor lidar frame
    vector<SensorFrame> imu_buffer_;                                     // record raw imu datas
    vector<SensorFrame> chassis_buffer_;                                 // record raw chassis datas
    vector<pair<LidarFrame, SensorFrame>> aligned_lidar_imu_buffer_;     // aligned lidar frame and interpolated imu attitude at lidar stamp
    vector<pair<LidarFrame, SensorFrame>> aligned_lidar_chassis_buffer_; // aligned lidar frame and interpolated imu attitude at lidar stamp
    void setInitT_l_m();

    Frame f_l_i;
    Frame f_l_c;
    Eigen::Quaterniond t_l_b_;
    Eigen::Quaterniond t_l_v_; // result
    Eigen::Matrix4d T_l_m;
    // Eigen::Vector3d flag{0.0, 0.0, 0.0};

    CloudT::Ptr local_map_{nullptr};
    pcl::VoxelGrid<PointT> downer_;                                               // downsample local map
    pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr register_{nullptr}; // register object

    vector<pair<Frame, Frame>> corres1_;
    vector<pair<Frame, Frame>> corres2_; // 点云优化后

    vector<EigenAffineVector> corres2affine(vector<pair<Frame, Frame>> corres);

    void getAlignedBuffer(vector<SensorFrame> sensor_buffer_, string sensorName);

    vector<pair<Frame, Frame>> alignedBuffer2corres(vector<pair<LidarFrame, SensorFrame>> aligned_sensor_buffer_);

    Eigen::Affine3d frame2affine(Frame frame);

    void printFrame(Frame frame);

    void savePoseKalibr();
    
    void savePoseEE();

    void saveCombinedMap(string sensorName, string fileName, vector<pair<LidarFrame, SensorFrame>> aligned_sensor_buffer_);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};