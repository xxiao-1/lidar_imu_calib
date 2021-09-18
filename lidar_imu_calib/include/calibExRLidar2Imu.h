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

struct ImuFrame
{
    double stamp;
    Eigen::Quaterniond rot;
    Eigen::Vector3d tra;
};

struct ChassisFrame
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
    void addImuFrame(const ImuFrame &data);

    //@brief: add chassis data and cache
    void addChassisFrame(const ChassisFrame &data);

    //@brief: integration imu data, align lidar odom and imu
    Eigen::Vector3d calibLidar2Imu();

    //@brief:  align lidar odom and chassis
    Eigen::Vector3d calibLidar2Chassis();

    //@brief: align chassis odom and imu
    // Eigen::Vector3d calibChassis2Imu(bool integration = false);

private:
    //@brief: interpolated attitude from start attitude to end attitude by scale
    Eigen::Quaterniond getInterpolatedAttitude(const Eigen::Quaterniond &q_s_w, const Eigen::Quaterniond &q_e_w, double scale);
 
    Eigen::Vector3d getInterpolatedTranslation(const Eigen::Vector3d &t_s_w, const Eigen::Vector3d &t_e_w, double scale);
    
    Eigen::Matrix3d skew(Eigen::Vector3d u);
    //@brief: update relative transform between neighbor lidar frame by aligned imu data
    void optimize();

    //@brief: update relative transform between neighbor lidar frame by aligned imu data
    void optimizeLidar2Chassis();

    //@brief: solve least square answer by constraints
    Eigen::Quaterniond solve(const vector<pair<Frame, Frame>> &corres);

    Eigen::Matrix4d solveX(const vector<pair<Frame, Frame>> &corres);

    Eigen::Vector3d init_R_{0.0, 0.0, 0.0};
    CloudT::Ptr last_lidar_cloud_{nullptr};
    vector<LidarFrame> lidar_buffer_;                                       // record relative transform between neighbor lidar frame
    vector<ImuFrame> imu_buffer_;                                            // record raw imu datas
    vector<ChassisFrame> chassis_buffer_;                                    // record raw chassis datas
    vector<pair<LidarFrame, ImuFrame>> aligned_lidar_imu_buffer_; // aligned lidar frame and interpolated imu attitude at lidar stamp
    vector<pair<LidarFrame, ChassisFrame>> aligned_lidar_chassis_buffer_; // aligned lidar frame and interpolated imu attitude at lidar stamp
    Eigen::Quaterniond q_l_b_;  
    Eigen::Quaterniond q_l_v_;                                                // result
    Eigen::Quaterniond t_l_b_;  
    Eigen::Quaterniond t_l_v_;                                                // result


    CloudT::Ptr local_map_{nullptr};                                              // local map
    pcl::VoxelGrid<PointT> downer_;                                               // downsample local map
    pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr register_{nullptr}; // register object

    vector<pair<Frame, Frame>> corres1_;
    vector<pair<Frame, Frame>> corres2_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};