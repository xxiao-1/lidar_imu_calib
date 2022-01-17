#include "calibExRLidar2Imu.h"
#include "HandEyeCalibration.h"
#include <omp.h>
#include <utility>
#include <tf/tf.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/LU>
#include <Eigen/QR>
#include <stdlib.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include "lidar_imu_calib/chassis_data.h"
#include "utils.h"
//for output
#include <fstream>
#include <iostream>

#define USE_SCAN_2_MAP true

// const
std::string imu_origin_fileName = "/home/xxiao/HitLidarImu/result/i0_imu_origin.txt";
// std::string data_set="parkinglot1";
std::string data_set = "small_fast";

CalibExRLidarImu::CalibExRLidarImu()
{
    imu_buffer_.clear();
    chassis_buffer_.clear();

    // init downsample object
    downer_.setLeafSize(0.7, 0.7, 0.7);

    // init register object
    register_.reset(new pclomp::NormalDistributionsTransform<PointT, PointT>());
    register_->setResolution(1.0);
    int avalib_cpus = omp_get_max_threads();
    register_->setNumThreads(avalib_cpus);
    register_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
}

CalibExRLidarImu::~CalibExRLidarImu()
{
}

void CalibExRLidarImu::setInitExR(Eigen::Vector3d init_R)
{
    init_R_ = init_R;
}

void CalibExRLidarImu::setInitT_l_m()
{
    T_l_m = Eigen::Matrix4d::Identity();
}

// 每条消息都要执行一次
void CalibExRLidarImu::addLidarData(const LidarData &data)
{
    if (!data.cloud || data.cloud->size() == 0)
    {
        cout << "no cloud in lidar data !!!" << endl;
        return;
    }

    if (!register_)
    {
        cout << "register no initialize !!!" << endl;
        return;
    }

    //downsample lidar cloud for save align time
    CloudT::Ptr downed_cloud(new CloudT);
    downer_.setInputCloud(data.cloud);
    downer_.filter(*downed_cloud);

    if (!local_map_)
    {
        local_map_.reset(new CloudT);
        *local_map_ += *(data.cloud);

        LidarFrame frame;
        frame.stamp = data.stamp;
        frame.T = Eigen::Matrix4d::Identity();
        frame.gT = Eigen::Matrix4d::Identity();
        frame.cloud = downed_cloud;
        lidar_buffer_.push_back(move(frame));
        setInitT_l_m();
        return;
    }

    // downsample local map for save align time
    CloudT::Ptr downed_map(new CloudT);
    downer_.setInputCloud(local_map_);
    downer_.filter(*downed_map);
    local_map_ = downed_map;

    // get transform between frame and local map
    register_->setInputSource(downed_cloud);
    register_->setInputTarget(local_map_);
    CloudT::Ptr aligned(new CloudT);
    register_->align(*aligned, T_l_m.cast<float>());

    if (!register_->hasConverged())
    {
        cout << "register cant converge, please check initial value !!!" << endl;
        return;
    }
    T_l_m = (register_->getFinalTransformation()).cast<double>();

    // generate lidar frame
    LidarFrame frame;
    frame.stamp = data.stamp;
    frame.gT = T_l_m;
    Eigen::Matrix4d last_T_l_m = lidar_buffer_.back().gT;
    frame.T = last_T_l_m.inverse() * T_l_m; //T:k+1相对于k的变化矩阵 t = k+1, gT: 相对于初值的变化矩阵
    frame.cloud = downed_cloud;
    lidar_buffer_.push_back(move(frame)); // 地图逐渐形成，匹配用的是法向量，不是ICP

    // update local map
    *local_map_ += *aligned; // 最后再加入地图中
}

// imu odometry
void CalibExRLidarImu::addImuFrame(const vector<ImuData> &imu_raw_buffer)
{
    Eigen::Quaterniond initRot = imu_raw_buffer[0].rot;
    Eigen::Vector3d last_P(0, 0, 0);
    Eigen::Vector3d last_V(0, 0, 0);
    double deltaT = 0;
    bool timeFlag = true;
    double initTime = imu_raw_buffer[0].stamp;

    ofstream imu_origin_file;
    deleteFile(imu_origin_fileName);
    imu_origin_file.open(imu_origin_fileName, ios::app);
    imu_origin_file.precision(10);
    double scale = 1.0;
    for (int i = 0; i < imu_raw_buffer.size(); ++i)
    {
        ImuData imupose1 = imu_raw_buffer[i];
        imupose1.rot = initRot.inverse() * imupose1.rot;

        if (i >= 1)
        {
            ImuData imupose = imu_raw_buffer[i - 1];
            deltaT = imupose1.stamp - imupose.stamp;

            // use ori
            double deltaT = imupose1.stamp - initTime;
            // if (data_set == "parkinglot1")
            // {
            //     if (deltaT > 85)
            //     {
            //         scale = 1.0 / 2560;
            //     }
            //     else if (deltaT > 70)
            //     {
            //         scale = 1.0 / 1280.0;
            //     }
            //     else if (deltaT > 50)
            //     {
            //         scale = 1.0 / 640;
            //     }
            //     else if (deltaT > 16)
            //     {
            //         scale = 1.0 / 360;
            //     }
            //     else if (deltaT > 10)
            //     {
            //         scale = 1.0 / 70;
            //     }
            //     else if (deltaT > 7)
            //     {
            //         scale = 1.0 / 1.05;
            //     }

                if (deltaT > 3 && timeFlag)
                {
                    last_V = last_V / 1.15;
                    timeFlag = false;
                }
            // }
            // else if (data_set == "small_fast")
            // {
                if (deltaT > 6)
            {
                scale=1.0/1280;
            }
            if (deltaT > 5)
            {
                scale=1.0/640;
            }
            if (deltaT > 4)
            {
                scale = 1.0 / 360;
            }
            if (deltaT > 2)
            {
                scale = 1.0 / 240;
            }

            imupose.rot = initRot.inverse() * imupose.rot;
            Eigen::Vector3d averageAcc = scale * (imupose.acc + imupose1.acc) / 2.0;
            last_P = last_P + imupose.rot * last_V * deltaT + imupose.rot * (averageAcc / 2.0 * deltaT * deltaT);
            last_V = (imupose.rot.inverse() * imupose1.rot) * averageAcc * deltaT + last_V;
        }
        Eigen::Vector3d tmpP = last_P;
        // if (data_set == "parkinglot1")
        // {
        //     tmpP = last_P * 1e-7;
        // }
        // else if (data_set == "small_fast")
        // {
        tmpP = last_P * 180 / 6605541.0;
        //     std::cout<<"--2"<<std::endl;
        // }

        // std::cout<<"tmpP="<<tmpP.transpose()<<std::endl;

        // 储存imu frame
        SensorFrame imuFrame;
        imuFrame.stamp = imupose1.stamp;
        imuFrame.tra = tmpP;
        imuFrame.rot = imupose1.rot;
        imu_buffer_.push_back(imuFrame);

        //存储位姿
        imu_origin_file << imupose1.stamp << " "
                        << tmpP[0] << " "
                        << tmpP[1] << " "
                        << tmpP[2] << " "
                        << imupose1.rot.x() << " "
                        << imupose1.rot.y() << " "
                        << imupose1.rot.z() << " "
                        << imupose1.rot.w();
        imu_origin_file << "\n";
    }
    imu_origin_file.close();
}

void CalibExRLidarImu::addChassisFrame(const SensorFrame &data)
{
    chassis_buffer_.push_back(data);
}

// 姿态插值
Eigen::Quaterniond CalibExRLidarImu::getInterpolatedAttitude(const Eigen::Quaterniond &q_s_w, const Eigen::Quaterniond &q_e_w, double scale)
{
    if (scale < 0 || scale > 1)
        return move(Eigen::Quaterniond().Identity());

    // calculate angleaxis difference
    Eigen::Quaterniond q_e_s = q_s_w.inverse() * q_e_w;
    q_e_s.normalize();
    Eigen::AngleAxisd diff_angle_axis(q_e_s);

    // interpolated attitude by scale
    double interpolated_angle = diff_angle_axis.angle() * scale;
    Eigen::Quaterniond q_ie_s(Eigen::AngleAxisd(interpolated_angle, diff_angle_axis.axis()).toRotationMatrix());
    Eigen::Quaterniond q_ie_w = q_s_w * q_ie_s;
    q_ie_w.normalize();

    return move(q_ie_w);
}

// 位移插值
Eigen::Vector3d CalibExRLidarImu::getInterpolatedTranslation(const Eigen::Vector3d &t_s_w, const Eigen::Vector3d &t_e_w, double scale)
{
    if (scale < 0 || scale > 1)
        return move(Eigen::Vector3d(0));

    Eigen::Vector3d t_ie_w = (t_e_w - t_s_w) * scale + t_s_w;

    return move(t_ie_w);
}

Eigen::Matrix3d CalibExRLidarImu::skew(Eigen::Vector3d u)
{
    Eigen::Matrix3d u_hat = Eigen::MatrixXd::Zero(3, 3);
    u_hat(0, 1) = u(2);
    u_hat(1, 0) = -u(2);
    u_hat(0, 2) = -u(1);
    u_hat(2, 0) = u(1);
    u_hat(1, 2) = u(0);
    u_hat(2, 1) = -u(0);

    return u_hat;
}

Eigen::Matrix4d CalibExRLidarImu::solveX(const vector<pair<Frame, Frame>> &corres)
{

    if (corres.size() == 0)
    {
        cout << "no constraint found !!!" << endl;
        return move(Eigen::Matrix4d::Identity());
    }

    cout << "constraints size " << corres.size() << endl;

    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(12 * corres.size(), 12);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(12 * corres.size());
    for (int i = 0; i < corres.size(); i++)
    {
        //extract R,t from homogophy matrix
        // TODO: AX=XB 是A到B的
        Frame B = corres[i].first;
        Frame A = corres[i].second;
        //extract R,t from homogophy matrix
        Eigen::Matrix3d Ra = A.rot.normalized().toRotationMatrix();
        Eigen::Vector3d Ta = A.tra;
        Eigen::Matrix3d Rb = B.rot.normalized().toRotationMatrix();
        Eigen::Vector3d Tb = B.tra;

        m.block<9, 9>(12 * i, 0) = Eigen::MatrixXd::Identity(9, 9) - Eigen::kroneckerProduct(Ra, Rb);
        Eigen::Matrix3d Ta_skew = skew(Ta);
        m.block<3, 9>(12 * i + 9, 0) = Eigen::kroneckerProduct(Ta_skew, Tb.transpose());
        m.block<3, 3>(12 * i + 9, 9) = Ta_skew - Ta_skew * Ra;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeFullV | Eigen::ComputeFullU);
    // CHECK(svd.computeV()) << "fail to compute V";

    Eigen::Matrix3d R_alpha;
    R_alpha.row(0) = svd.matrixV().block<3, 1>(0, 11).transpose();
    R_alpha.row(1) = svd.matrixV().block<3, 1>(3, 11).transpose();
    R_alpha.row(2) = svd.matrixV().block<3, 1>(6, 11).transpose();
    //double a = std::fabs(R_alpha.determinant());
    //double alpha = R_alpha.determinant()/(pow(std::fabs(R_alpha.determinant()),4./3.));
    double det = R_alpha.determinant();
    double alpha = std::pow(std::abs(det), 4. / 3.) / det;
    Eigen::HouseholderQR<Eigen::Matrix3d> qr(R_alpha / alpha);

    Eigen::Matrix4d handeyetransformation = Eigen::Matrix4d ::Identity(4, 4);
    Eigen::Matrix3d Q = qr.householderQ();
    Eigen::Matrix3d Rwithscale = alpha * Q.transpose() * R_alpha;
    Eigen::Vector3d R_diagonal = Rwithscale.diagonal();
    for (int i = 0; i < 3; i++)
    {
        handeyetransformation.block<3, 1>(0, i) = int(R_diagonal(i) >= 0 ? 1 : -1) * Q.col(i);
    }

    handeyetransformation.topRightCorner(3, 1) = svd.matrixV().block<3, 1>(9, 11) / alpha;
    return handeyetransformation;
}

// solve函数 lidar 2 imu
Frame CalibExRLidarImu::solve(const vector<pair<Frame, Frame>> &corres)
{
    Frame result;
    if (corres.size() == 0)
    {
        result.rot = Eigen::Quaterniond().Identity();
        result.tra = Eigen::MatrixXd::Zero(3, 1);
        cout << "no constraint found !!!" << endl;
        return result;
    }

    cout << "constraints size " << corres.size() << endl;

    // transform quaternion to skew symmetric matrix
    // 四元数到斜对称矩阵的变换
    auto toSkewSymmetric = [](const Eigen::Vector3d &q) -> Eigen::Matrix3d
    {
        Eigen::Matrix3d mat = Eigen::Matrix3d::Zero();
        mat(0, 1) = -q.z();
        mat(0, 2) = q.y();
        mat(1, 0) = q.z();
        mat(1, 2) = -q.x();
        mat(2, 0) = -q.y();
        mat(2, 1) = q.x();

        return move(mat);
    };

    // create homogeneous linear equations
    // 创建齐次线性方程组
    Eigen::MatrixXd A(corres.size() * 4, 4);
    for (int i = 0; i < corres.size(); i++)
    {
        // get relative transform
        const auto &q_l2_l1 = corres[i].first.rot;
        const auto &q_b2_b1 = corres[i].second.rot;

        // get left product matrix
        Eigen::Vector3d q_b2_b1_vec = q_b2_b1.vec();
        Eigen::Matrix4d left_Q_b2_b1 = Eigen::Matrix4d::Zero();
        left_Q_b2_b1.block<1, 3>(0, 1) = -q_b2_b1_vec.transpose();
        left_Q_b2_b1.block<3, 1>(1, 0) = q_b2_b1_vec;
        left_Q_b2_b1.block<3, 3>(1, 1) = toSkewSymmetric(q_b2_b1_vec);
        left_Q_b2_b1 += q_b2_b1.w() * Eigen::Matrix4d::Identity();

        // get right product matrix
        Eigen::Vector3d q_l2_l1_vec = q_l2_l1.vec();
        Eigen::Matrix4d right_Q_l2_l1 = Eigen::Matrix4d::Zero();
        right_Q_l2_l1.block<1, 3>(0, 1) = -q_l2_l1_vec.transpose();
        right_Q_l2_l1.block<3, 1>(1, 0) = q_l2_l1_vec;
        right_Q_l2_l1.block<3, 3>(1, 1) = -toSkewSymmetric(q_l2_l1_vec);
        right_Q_l2_l1 += q_l2_l1.w() * Eigen::Matrix4d::Identity();

        // add loss function
        double angle_distance = 180.0 / M_PI * q_b2_b1.angularDistance(q_l2_l1);
        double huber = angle_distance > 2.0 ? 2.0 / angle_distance : 1.0;

        A.block<4, 4>(i * 4, 0) = huber * (left_Q_b2_b1 - right_Q_l2_l1);
    }

    // 奇异值分解法求解齐次线性方程组
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Eigen::Quaterniond q_l_b(x(0), x(1), x(2), x(3));
    Eigen::MatrixXd R_l_b = q_l_b.normalized().toRotationMatrix();

    // t
    // Eigen::MatrixXd A1(corres.size() * 3, 3);
    // Eigen::MatrixXd B1(corres.size() * 3, 1);
    // for (int i = 0; i < corres.size(); i++)
    // {
    //     // get relative transform
    //     Eigen::Vector3d t_l2_l1 = corres[i].first.tra;
    //     Eigen::Vector3d t_b2_b1 = corres[i].second.tra;
    //     Eigen::Quaterniond q_b2_b1 = corres[i].second.rot;

    //     A1.block<3, 3>(i, 0) = Eigen::MatrixXd::Identity(3, 3) - q_b2_b1.normalized().toRotationMatrix();
    //     B1.block<3, 1>(i, 0) = corres[i].second.tra - R_l_b * corres[i].first.tra;
    // }

    // Eigen::MatrixXd t = A1.colPivHouseholderQr().solve(B1);
    // std::cout << "tras is" << t.transpose() << std::endl;

    Eigen::Vector3d t(0, 0, 0);

    result.rot = q_l_b.normalized();
    Eigen::Vector3d ang = toEulerAngle(result.rot);
    if (abs(ang[2]) - 180 < 10 && abs(ang[2]) - 180 > -10)
    {
        Eigen::AngleAxisd rotation_vector(M_PI, Eigen::Vector3d(0, 0, 1));
        Eigen::Quaterniond oriQ = Eigen::Quaterniond(rotation_vector);
        result.rot = oriQ * result.rot;
    }
    result.tra = t;
    return result;
}

/*
  优化solve结果,一次性执行所有点云
*/
void CalibExRLidarImu::optimize(string sensorName, vector<pair<LidarFrame, SensorFrame>> &aligned_sensor_buffer_)
{
    Eigen::Quaterniond q_l_b_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_l_b = {0, 0, 0};

    if (sensorName == "imu")
    {
        q_l_b_ = f_l_i.rot;
        t_l_b = f_l_i.tra;
    }
    else if (sensorName == "chassis")
    {
        q_l_b_ = f_l_c.rot;
        t_l_b = f_l_c.tra;
    }

    if (aligned_sensor_buffer_.size() == 0 || !register_)
    {
        cout << "no aligned data or register !!!" << endl;
        return;
    }

    // clear local map and initialize
    if (local_map_)
        local_map_->clear();
    else
        local_map_.reset(new CloudT);
    *local_map_ += *(aligned_sensor_buffer_[0].first.cloud);

    // use scan2match with estimated initial value to update lidar frame
    for (int i = 1; i < aligned_sensor_buffer_.size(); i++)
    {
        // get front and back frames
        auto &aligned1 = aligned_sensor_buffer_[i - 1];
        auto &aligned2 = aligned_sensor_buffer_[i];

        // downsample local map and lidar cloud for save align time
        CloudT::Ptr downed_map(new CloudT);
        downer_.setInputCloud(local_map_);
        downer_.filter(*downed_map);
        local_map_ = downed_map;

        // calculate estimated T_l_m
        Eigen::Matrix3d R_l1_m = aligned1.first.gT.block<3, 3>(0, 0);
        Eigen::Quaterniond q_b1_w = aligned1.second.rot;
        Eigen::Quaterniond q_b2_w = aligned2.second.rot;
        Eigen::Quaterniond est_q_b2_b1 = q_b1_w.inverse() * q_b2_w;

        Eigen::Matrix3d est_R_l2_l1 = Eigen::Matrix3d(q_l_b_.inverse() * est_q_b2_b1 * q_l_b_); // 用imu变化求lidar的变化
        Eigen::Matrix3d est_R_l2_m = R_l1_m * est_R_l2_l1;
        Eigen::Matrix4d est_T_l2_m = Eigen::Matrix4d::Identity();
        est_T_l2_m.block<3, 3>(0, 0) = est_R_l2_m;
        est_T_l2_m.block<3, 1>(0, 3) = aligned1.first.gT.block<3, 1>(0, 3) + Eigen::Matrix3d(q_l_b_) * (aligned2.second.tra - aligned1.second.tra);

        // register
        register_->setInputSource(aligned2.first.cloud);
        register_->setInputTarget(local_map_);
        CloudT::Ptr aligned(new CloudT);
        register_->align(*aligned, est_T_l2_m.cast<float>()); // 相当于多了一个初值
        if (!register_->hasConverged())
        {
            cout << "register cant converge, please check initial value !!!" << endl;
            return;
        }
        Eigen::Matrix4d T_l2_m = (register_->getFinalTransformation()).cast<double>();

        // update lidar frame
        aligned2.first.gT = T_l2_m;
        Eigen::Matrix4d T_l1_m = aligned1.first.gT;
        aligned2.first.T = T_l1_m.inverse() * T_l2_m;

        CloudT::Ptr changed(new CloudT);
        pcl::transformPointCloud(*aligned2.first.cloud, *changed, T_l2_m);
        // update local map
        *local_map_ += *changed;
    }
    std::cout << "transformed" << sensorName << std::endl;
    // int ret = pcl::io::savePCDFileBinary("/home/xxiao/HitLidarImu/result/" + sensorName + "localMap.pcd", *local_map_);
    // std::cout << "combined cloud saved" << ret << std::endl;

    // generate constraints
    vector<pair<Frame, Frame>> corres = alignedBuffer2corres(aligned_sensor_buffer_);
    if (sensorName == "imu")
    {
        f_l_i = solve(corres);
    }
    else if (sensorName == "chassis")
    {
        f_l_c = solve(corres);
    }

    // // check whether optimize fail
    // double angle = fabs(f_l_c.rot.angularDistance(q));
    // if (angle > 0.5236)
    // {
    //     cout << "the difference between before and after optimze is " << angle << " which greater than given threshold 0.5236 !!!" << endl;
    //     return;
    // }
}

void CalibExRLidarImu::calibLidar2Imu()
{
    getAlignedBuffer(imu_buffer_, "imu");
    vector<pair<Frame, Frame>> corres = alignedBuffer2corres(aligned_lidar_imu_buffer_);
    vector<EigenAffineVector> t = corres2affine(corres);
    EigenAffineVector t1 = t[0];
    EigenAffineVector t2 = t[1];

    // 手动真值
    f_l_i.rot = yawToQuaterniond(-30);
    Eigen::Vector3d ang1 = toEulerAngle(f_l_i.rot);
    savePose("i1_imu_gT", f_l_i, aligned_lidar_imu_buffer_);
    return;
    // 优化
    // optimize("imu", aligned_lidar_imu_buffer_);
    // savePose("imu_ngT", f_l_i, aligned_lidar_imu_buffer_);

    // 线性solve
    f_l_i = solve(corres);
    savePose("i2_imu_linear", f_l_i, aligned_lidar_imu_buffer_);
    Eigen::Vector3d ang2 = toEulerAngle(f_l_i.rot);

    // double
    camodocal::HandEyeCalibration ceresHandeye;
    auto a_l_i = ceresHandeye.solveCeres(t1, t2, f_l_i.rot, f_l_i.tra);
    f_l_c.rot = Eigen::Quaternion<double>(a_l_i.rotation());
    f_l_c.tra = Eigen::Matrix<double, 3, 1>(a_l_i.translation());
    savePose("i3_imu_double", f_l_i, aligned_lidar_imu_buffer_);
    Eigen::Vector3d ang3 = toEulerAngle(f_l_i.rot);

    // f_l_i.rot = toEulerAngle(90.834);
    // savePose("i4_imu_multi", f_l_i, aligned_lidar_imu_buffer_);
    // Eigen::Vector3d ang4 = toEulerAngle(f_l_i.rot);

    std::cout << "imu_gT=" << ang1.transpose() << std::endl;
    std::cout << "imu_linear=" << ang2.transpose() << std::endl;
    std::cout << "imu_double=" << ang3.transpose() << std::endl;
    // std::cout << "imu_multi=" << ang4.transpose() << std::endl;
}

void CalibExRLidarImu::calibLidar2Chassis()
{

    // lidar chassis
    getAlignedBuffer(chassis_buffer_, "chassis");
    vector<pair<Frame, Frame>> corres = alignedBuffer2corres(aligned_lidar_chassis_buffer_);
    vector<EigenAffineVector> t = corres2affine(corres);
    EigenAffineVector t1 = t[0];
    EigenAffineVector t2 = t[1];

    // 手动真值
    f_l_c.rot = yawToQuaterniond(2.5);
    Eigen::Vector3d ang11 = toEulerAngle(f_l_c.rot);
    savePose("c1_chassis_gT", f_l_c, aligned_lidar_chassis_buffer_);
    savePoseKalibr();
    return;
    // 线性solve
    f_l_c = solve(corres);
    Eigen::Vector3d ang22 = toEulerAngle(f_l_c.rot);
    savePose("c2_chassis_linear", f_l_c, aligned_lidar_chassis_buffer_);

    // double
    camodocal::HandEyeCalibration ceresHandeye;
    auto a_l_c = ceresHandeye.solveCeres(t1, t2, f_l_c.rot, f_l_c.tra);
    f_l_c.rot = Eigen::Quaternion<double>(a_l_c.rotation());
    f_l_c.tra = Eigen::Matrix<double, 3, 1>(a_l_c.translation());
    Eigen::Vector3d ang33 = toEulerAngle(f_l_c.rot);
    savePose("c3_chassis_ceres", f_l_c, aligned_lidar_chassis_buffer_);

    // std::cout << "chassis_yaw_gT=" << -2.5 << std::endl;
    std::cout << "chassis_gT=" << ang11.transpose() << std::endl;
    std::cout << "chassis_linear=" << ang22.transpose() << std::endl;
    std::cout << "chassis_ceres=" << ang33.transpose() << std::endl;
}

void CalibExRLidarImu::calibImu2Chassis()
{

    // lidar chassis
    getAlignedBuffer(imu_buffer_, "imu");
    vector<pair<Frame, Frame>> corres_li = alignedBuffer2corres(aligned_lidar_imu_buffer_);
    getAlignedBuffer(chassis_buffer_, "chassis");
    vector<pair<Frame, Frame>> corres_lc = alignedBuffer2corres(aligned_lidar_chassis_buffer_);

    int size = corres_lc.size();
    if (corres_li.size() < size)
    {
        size = corres_li.size();
    }
    vector<pair<Frame, Frame>> corres;
    for (int i = 0; i < size; i++)
    {
        Frame imuF = corres_li[i].second;
        Frame chassisF = corres_lc[i].second;
        corres.push_back(move(pair<Frame, Frame>(imuF, chassisF)));
    }

    // 统一solve
    f_i_c = solve(corres);
    std::cout << "1--线性求解结果" << std::endl;
    Eigen::Vector3d ang22 = toEulerAngle(f_i_c.rot);

    vector<EigenAffineVector> t = corres2affine(corres);
    EigenAffineVector t1 = t[0];
    EigenAffineVector t2 = t[1];

    camodocal::HandEyeCalibration ceresHandeye;
    auto a_l_c = ceresHandeye.solveCeres(t1, t2, f_i_c.rot, f_i_c.tra);
    f_i_c.rot = Eigen::Quaternion<double>(a_l_c.rotation());
    f_i_c.tra = Eigen::Matrix<double, 3, 1>(a_l_c.translation());
    std::cout << "3--optimize结果" << std::endl;
    std::cout << a_l_c.matrix() << std::endl;

    Eigen::Vector3d ang33 = toEulerAngle(f_i_c.rot);

    std::cout << "imu2chassis_gT=" << 144.5 << std::endl;
    std::cout << "imu2chassis_linear=" << ang22.transpose() << std::endl;
    std::cout << "imu2chassis_ceres=" << ang33.transpose() << std::endl;
}

void CalibExRLidarImu::calibMulti()
{
    // 获取位姿对
    getAlignedBuffer(imu_buffer_, "imu");
    vector<pair<Frame, Frame>> corres_li = alignedBuffer2corres(aligned_lidar_imu_buffer_);
    getAlignedBuffer(chassis_buffer_, "chassis");
    vector<pair<Frame, Frame>> corres_lc = alignedBuffer2corres(aligned_lidar_chassis_buffer_);

    // 获得初值
    f_l_i = solve(corres_li);
    std::cout << "1------------------lidar2imu--线性求解结果" << std::endl;
    printFrame(f_l_i);
    savePose("imu_linear", f_l_i, aligned_lidar_imu_buffer_);
    // saveCombinedMap("imu", "1_imu", aligned_lidar_imu_buffer_);

    // optimize("imu", aligned_lidar_imu_buffer_);
    // std::cout << "2------------------lidar2imu--优化结果" << std::endl;
    // printFrame(f_l_i);
    // saveCombinedMap("imu", "2_imu", aligned_lidar_imu_buffer_);

    f_l_c = solve(corres_lc);
    std::cout << "1=================lidar2chassis--线性求解结果" << std::endl;
    printFrame(f_l_c);
    savePose("chassis_linear", f_l_c, aligned_lidar_chassis_buffer_);
    // saveCombinedMap("chassis", "1_chassis", aligned_lidar_chassis_buffer_);

    // optimize("chassis", aligned_lidar_chassis_buffer_);
    // std::cout << "2=================lidar2chassis--优化结果" << std::endl;
    // printFrame(f_l_c);
    // saveCombinedMap("chassis", "2_chassis", aligned_lidar_chassis_buffer_);

    Eigen::Affine3d a12, a31, a23;
    a12 = frame2affine(f_l_i);
    Eigen::Transform<double, 3, Eigen::Affine> a13 = frame2affine(f_l_c);
    a31 = a13.inverse();
    a23 = a12.inverse() * a13;
    std::cout << "imu2chassis" << std::endl;
    std::cout << a23.matrix() << std::endl;

    std::cout << "---------------------------------------------------------------------------------" << std::endl;

    // 格式转化
    vector<EigenAffineVector> t_li = corres2affine(corres_li);
    EigenAffineVector t1 = t_li[0];
    EigenAffineVector t2 = t_li[1];

    vector<EigenAffineVector> t_lv = corres2affine(corres_lc);
    EigenAffineVector t3 = t_lv[1];

    camodocal::HandEyeCalibration ceresHandeye;

    std::vector<Eigen::Affine3d> aList = ceresHandeye.solveMultiCeres(t1, t2, t3, a12, a31, a23);

    Eigen::Affine3d a_l_i = aList[0];
    f_l_i.rot = Eigen::Quaternion<double>(a_l_i.rotation());
    f_l_i.tra = Eigen::Matrix<double, 3, 1>(a_l_i.translation());

    Eigen::Affine3d a_l_c = aList[1].inverse();
    f_l_c.rot = Eigen::Quaternion<double>(a_l_c.rotation());
    f_l_c.tra = Eigen::Matrix<double, 3, 1>(a_l_c.translation());

    savePose("imu_ceres", f_l_i, aligned_lidar_imu_buffer_);
    savePose("chassis_ceres", f_l_c, aligned_lidar_chassis_buffer_);

    // saveCombinedMap("imu", "3_imu", aligned_lidar_imu_buffer_);
    // saveCombinedMap("chassis", "3_chassis", aligned_lidar_chassis_buffer_);
}

vector<EigenAffineVector> CalibExRLidarImu::corres2affine(vector<pair<Frame, Frame>> corres)
{
    vector<EigenAffineVector> t;
    EigenAffineVector t1, t2;
    for (int i = 0; i < corres.size(); i++)
    {
        Eigen::Affine3d t1_T = frame2affine(corres[i].first);
        t1.push_back(t1_T);

        Eigen::Affine3d t2_T = frame2affine(corres[i].second);
        t2.push_back(t2_T);
    }
    t.push_back(t1);
    t.push_back(t2);
    return t;
}

Eigen::Affine3d CalibExRLidarImu::frame2affine(Frame frame)
{
    Eigen::Affine3d a = Eigen::Affine3d::Identity();
    a.rotate(frame.rot);
    a.translate(frame.tra);
    return a;
}

void CalibExRLidarImu::getAlignedBuffer(vector<SensorFrame> sensor_buffer_, string sensorName)
{
    if (lidar_buffer_.size() == 0 && sensor_buffer_.size() == 0)
    {
        cout << "no lidar data or chassis data !!!" << endl;
        return;
    }

    cout << "total lidar buffer size " << lidar_buffer_.size()
         << ", sensor buffer size " << sensor_buffer_.size() << endl;

    // move invalid lidar frame which got before first chassis frame
    auto invalid_lidar_it = lidar_buffer_.begin();
    for (; invalid_lidar_it != lidar_buffer_.end(); invalid_lidar_it++)
    {
        if (invalid_lidar_it->stamp >= sensor_buffer_[0].stamp)
            break;
    }
    if (invalid_lidar_it != lidar_buffer_.begin())
        lidar_buffer_.erase(lidar_buffer_.begin(), invalid_lidar_it);
    if (lidar_buffer_.size() == 0)
    {
        cout << "no valid lidar frame !!!" << endl;
        return;
    }

    // get time-aligned lidar odometry rotation and chassis integration rotation
    auto last_sensor_it = sensor_buffer_.begin();
    for (int i = 0; i < lidar_buffer_.size(); i++)
    {
        // get lidar information
        const auto &lidar_frame = lidar_buffer_[i];

        // get last imu frame which before current lidar frame
        for (; last_sensor_it != sensor_buffer_.end(); last_sensor_it++)
        {
            if (last_sensor_it->stamp >= lidar_frame.stamp)
                break;
        }
        if (last_sensor_it != sensor_buffer_.begin())
            last_sensor_it--;

        // get interpolated imu attitude at lidar stamp
        auto s_it1 = last_sensor_it;
        auto s_it2 = last_sensor_it + 1;
        if (sensor_buffer_.end() == s_it2)
            break;
        assert(s_it2->stamp >= lidar_frame.stamp || s_it1->stamp < s_it2->stamp);
        double scale = (lidar_frame.stamp - s_it1->stamp) / (s_it2->stamp - s_it1->stamp);
        Eigen::Quaterniond q_b1_w = s_it1->rot;
        Eigen::Quaterniond q_b2_w = s_it2->rot;
        Eigen::Quaterniond q_inter_w = getInterpolatedAttitude(q_b1_w, q_b2_w, scale);

        Eigen::Vector3d t_b1_w = s_it1->tra;
        Eigen::Vector3d t_b2_w = s_it2->tra;
        Eigen::Vector3d t_inter_w = getInterpolatedTranslation(t_b1_w, t_b2_w, scale);
        // std::cout << "scale is " << scale << " begin=" << t_b1_w<< " end=" << t_b2_w<< " inter=" << t_inter_w << std::endl;

        SensorFrame sensor_frame;
        sensor_frame.stamp = lidar_frame.stamp;
        sensor_frame.rot = q_inter_w;
        sensor_frame.tra = t_inter_w;

        if (sensorName == "chassis")
        {

            aligned_lidar_chassis_buffer_.push_back(move(pair<LidarFrame, SensorFrame>(lidar_frame, sensor_frame)));
        }
        else if (sensorName == "imu")
        {

            aligned_lidar_imu_buffer_.push_back(move(pair<LidarFrame, SensorFrame>(lidar_frame, sensor_frame)));
        }
    }
    // std::cout << "mark before save lidar" << std::endl;

    // savePoseEE();
}

vector<pair<Frame, Frame>> CalibExRLidarImu::alignedBuffer2corres(vector<pair<LidarFrame, SensorFrame>> aligned_sensor_buffer_)
{
    // solve initial transform between lidar and sensor
    vector<pair<Frame, Frame>> corres(0);
    for (int i = 1; i < aligned_sensor_buffer_.size(); i++)
    {
        // get neighbor aligned frame
        const auto &aligned1 = aligned_sensor_buffer_[i - 1];
        const auto &aligned2 = aligned_sensor_buffer_[i];

        // get initial relative transform between neighbor lidar
        Eigen::Quaterniond q_l2_l1 = Eigen::Quaterniond(aligned_sensor_buffer_[i].first.T.block<3, 3>(0, 0));

        // calculate relative transform between neighbor interpolated imu
        Eigen::Quaterniond q_v1_w = aligned1.second.rot;
        Eigen::Quaterniond q_v2_w = aligned2.second.rot;
        Eigen::Quaterniond q_v2_v1 = q_v1_w.inverse() * q_v2_w;

        Frame s_frame;
        s_frame.rot = q_v2_v1;
        s_frame.tra = aligned2.second.tra - aligned1.second.tra;

        Frame l_frame;
        l_frame.rot = q_l2_l1;
        l_frame.tra = aligned2.first.T.block<3, 1>(0, 3).transpose();

        corres.push_back(move(pair<Frame, Frame>(l_frame, s_frame)));
        corres1_ = corres;
    }
    return corres;
}

Frame CalibExRLidarImu::getDetlaFrame(Frame f1, Frame f2)
{
    Frame frame;
    Eigen::Quaterniond q_1 = f1.rot;
    Eigen::Quaterniond q_2 = f2.rot;
    Eigen::Quaterniond q_2_1 = q_1.inverse() * q_2;

    frame.rot = q_2_1;
    frame.tra = f2.tra - f1.tra;
    frame.tra[2] = 0.0;
    return frame;
}

vector<pair<Frame, Frame>> CalibExRLidarImu::singleBuffer2corres(vector<Frame> buffer1, vector<Frame> buffer2)
{
    // solve initial transform between lidar and sensor
    vector<pair<Frame, Frame>> corres(0);
    std::cout << buffer1.size() << " " << buffer2.size() << std::endl;
    assert(buffer1.size() == buffer2.size());
    int length = buffer1.size();
    for (int i = 1; i < length - 1; i++)
    {
        Frame frame1 = getDetlaFrame(buffer1[i], buffer1[i + 1]);
        Frame frame2 = getDetlaFrame(buffer2[i], buffer2[i + 1]);
        corres.push_back(pair<Frame, Frame>(frame1, frame2));
    }
    return corres;
}
void CalibExRLidarImu::calibSimulateDouble(vector<Frame> buffer1, vector<Frame> buffer2, Eigen::Quaterniond gt, Eigen::Vector3d gtT)
{

    //线性求解
    vector<pair<Frame, Frame>> corres_12 = singleBuffer2corres(buffer1, buffer2);
    Frame f_1_2 = solve(corres_12);

    std::cout << "线性求解结果:" << std::endl;
    printFrame(f_1_2);
    // cout << "角度差为：" << 180.0 / M_PI * gt.angularDistance(f_1_2.rot) << endl;
    // cout << "位移差为" << (f_1_2.tra-gtT).norm() << endl;

    // 非线性 先估计初值，然后再优化
    vector<EigenAffineVector> t = corres2affine(corres_12);
    EigenAffineVector t1 = t[0];
    EigenAffineVector t2 = t[1];

    camodocal::HandEyeCalibration ceresHandeye;
    auto a_1_2 = ceresHandeye.solveCeres(t1, t2, f_1_2.rot, f_1_2.tra, gt, gtT);

    // f_1_2.rot = Eigen::Quaternion<double>(a_1_2.rotation());
    // f_1_2.tra = Eigen::Matrix<double, 3, 1>(a_1_2.translation());

    // std::cout << "ceres非线性优化结果" << std::endl;
    // std::cout << a_1_2.matrix() << std::endl;
    // cout << "角度差为：" << 180.0 / M_PI * gt.angularDistance(Eigen::Quaternion<double>(a_1_2.rotation())) << endl;
}
void CalibExRLidarImu::calibSimulateMulti(vector<Frame> buffer1, vector<Frame> buffer2, vector<Frame> buffer3)
{
    vector<pair<Frame, Frame>> corres_12 = singleBuffer2corres(buffer1, buffer2);
    vector<pair<Frame, Frame>> corres_13 = singleBuffer2corres(buffer1, buffer3);
    vector<EigenAffineVector> t_12 = corres2affine(corres_12);
    vector<EigenAffineVector> t_13 = corres2affine(corres_13);
    EigenAffineVector t1 = t_12[0];
    EigenAffineVector t2 = t_12[1];
    EigenAffineVector t3 = t_13[1];

    assert(corres_12.size() == corres_13.size());

    camodocal::HandEyeCalibration ceresHandeye;
    Eigen::Affine3d a12, a31, a23;
    std::vector<Eigen::Affine3d> aList = ceresHandeye.solveMultiCeres(t1, t2, t3, a12, a31, a23);

    // Eigen::Affine3d a_l_i = aList[0];
    // f_l_i.rot = Eigen::Quaternion<double>(a_l_i.rotation());
    // f_l_i.tra = Eigen::Matrix<double, 3, 1>(a_l_i.translation());

    // Eigen::Affine3d a_l_c = aList[1].inverse();
    // f_l_c.rot = Eigen::Quaternion<double>(a_l_c.rotation());
    // f_l_c.tra = Eigen::Matrix<double, 3, 1>(a_l_c.translation());
}
void CalibExRLidarImu::printFrame(Frame frame)
{
    Eigen::Affine3d a = frame2affine(frame);
    std::cout << a.matrix() << std::endl;
}

// for kalibr b-spline
void CalibExRLidarImu::savePoseKalibr()
{
    ofstream lidarKalibrFile;
    std::string lidarKalibrFileName = "/home/xxiao/HitLidarImu/result/forKalibr/poseLidarKalibr.txt";

    deleteFile(lidarKalibrFileName);
    lidarKalibrFile.open(lidarKalibrFileName, ios::app); //pose
    lidarKalibrFile.precision(10);
    for (int i = 0; i < aligned_lidar_imu_buffer_.size(); i++)
    {
        LidarFrame lidar = aligned_lidar_imu_buffer_[i].first;

        lidarKalibrFile << ros::Time().fromSec(lidar.stamp) << " ";
        Eigen::VectorXd v(6);
        v.head(3) = lidar.gT.topRightCorner<3, 1>();

        Eigen::Matrix3d C = lidar.gT.topLeftCorner<3, 3>();
        Eigen::Vector3d p;
        double tr = std::max(-1.0, std::min((C(0, 0) + C(1, 1) + C(2, 2) - 1.0) * 0.5, 1.0));
        double a = acos(tr);

        if (fabs(a) < 1e-14)
        {
            p = Eigen::Vector3d::Zero();
        }
        else
        {
            p[0] = (C(2, 1) - C(1, 2));
            p[1] = (C(0, 2) - C(2, 0));
            p[2] = (C(1, 0) - C(0, 1));
            double n2 = p.norm();
            if (fabs(n2) < 1e-14)
            {
                p = Eigen::Vector3d::Zero();
            }
            else
            {
                double scale = -a / n2;
                p = scale * p;
            }
        }

        v.tail(3) = p;

        lidarKalibrFile << v[0] << " " << v[1];
        lidarKalibrFile << " " << v[2] << " " << v[3] << " " << v[4] << " " << v[5];
        lidarKalibrFile << "\n";
    }
    lidarKalibrFile.close();
}

// for EEHandEye
void CalibExRLidarImu::savePoseEE()
{
    // lidar
    ofstream myfile;
    std::string rmseFile = "/home/xxiao/HitLidarImu/result/poseLidar.txt";
    if (access(rmseFile.c_str(), 0) == 0) //文件存在
    {
        if (remove(rmseFile.c_str()) == 0)
        {
            std::cout << "poseLidar update done :)" << std::endl;
        }
        else
        {
            std::cout << "poseLidar update failed :(" << std::endl;
        }
    }
    myfile.open(rmseFile, ios::app); //pose
    myfile.precision(10);
    for (int i = 0; i < aligned_lidar_imu_buffer_.size(); i++)
    {
        LidarFrame lidar = aligned_lidar_imu_buffer_[i].first;

        myfile << ros::Time().fromSec(lidar.stamp) << " ";
        Eigen::VectorXd v(3);
        v = lidar.gT.topRightCorner<3, 1>();

        Eigen::Matrix3d C = lidar.gT.topLeftCorner<3, 3>();
        Eigen::Quaterniond q(C);

        myfile << v[0] << " " << v[1];
        myfile << " " << v[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
        myfile << "\n";
    }
    myfile.close();

    ofstream myfile1;
    myfile1.open("/home/xxiao/HitLidarImu/result/poseImu.txt", ios::app); //pose
    myfile1.precision(10);
    for (int i = 0; i < aligned_lidar_imu_buffer_.size(); i++)
    {
        SensorFrame sensor = aligned_lidar_imu_buffer_[i].second;

        myfile1 << ros::Time().fromSec(sensor.stamp) << " ";

        myfile1 << sensor.tra[0] << " " << sensor.tra[1] << " " << sensor.tra[2] << " " << sensor.rot.x() << " " << sensor.rot.y() << " " << sensor.rot.z() << " " << sensor.rot.w() << "\n";
    }
    myfile1.close();

    ofstream myfile2;
    myfile2.open("/home/xxiao/HitLidarImu/result/poseWheel.txt", ios::app); //pose
    myfile2.precision(10);
    for (int i = 0; i < aligned_lidar_chassis_buffer_.size(); i++)
    {
        SensorFrame sensor = aligned_lidar_chassis_buffer_[i].second;

        myfile2 << ros::Time().fromSec(sensor.stamp) << " ";

        myfile2 << sensor.tra[0] << " " << sensor.tra[1] << " " << sensor.tra[2] << " " << sensor.rot.x() << " " << sensor.rot.y() << " " << sensor.rot.z() << " " << sensor.rot.w() << "\n";
    }
    myfile2.close();
}

void CalibExRLidarImu::savePose(string sensorName, Frame f_a_b, vector<pair<LidarFrame, SensorFrame>> aligned_sensor_buffer_)
{
    //get translation matrix
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(f_a_b.rot);
    Eigen::Vector3d tt(0, 0, 0);
    T.pretranslate(tt); // 把平移向量设成 (1,3,4)

    std::cout << "save pose of " << sensorName << " length is " << aligned_sensor_buffer_.size() << std::endl;
    ofstream lidar_file;
    std::string lidar_fileName = "/home/xxiao/HitLidarImu/result/lidar.txt";
    deleteFile(lidar_fileName);

    lidar_file.open(lidar_fileName, ios::app);
    lidar_file.precision(10);
    for (int i = 0; i < aligned_sensor_buffer_.size(); i++)
    {
        LidarFrame lidar = aligned_sensor_buffer_[i].first;
        if (true)
        {
            lidar_file << ros::Time().fromSec(lidar.stamp) << " ";
            Eigen::VectorXd v(3);
            v = lidar.gT.topRightCorner<3, 1>();
            // v = changeTrans(v, -1.7, 0.97);

            Eigen::Matrix3d C = lidar.gT.topLeftCorner<3, 3>();
            Eigen::Quaterniond q1(C);
            Eigen::Quaterniond q = clearRollPitch(q1);
            // v[2] = 0;
            lidar_file << v[0] << " " << v[1]
                       << " " << v[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
            lidar_file << "\n";
        }
    }
    lidar_file.close();

    ofstream sensor_file;
    std::string sensor_fileName = "/home/xxiao/HitLidarImu/result/" + sensorName + ".txt";
    deleteFile(sensor_fileName);

    sensor_file.open(sensor_fileName, ios::app);
    sensor_file.precision(10);
    double time_offset = 0.0;

    for (int i = 0; i < aligned_sensor_buffer_.size(); i++)
    {
        SensorFrame sensor = aligned_sensor_buffer_[i].second;
        sensor_file << ros::Time().fromSec(sensor.stamp - time_offset) << " ";
        Frame s;
        s.tra = sensor.tra;
        s.rot = sensor.rot;

        // translation
        Eigen::Vector3d vs(s.tra);
        vs = T * vs;
        Eigen::Quaterniond qs(s.rot);
        sensor_file
            << vs[0] << " " << vs[1]
            << " " << vs[2] << " " << qs.x() << " " << qs.y() << " " << qs.z() << " " << qs.w();
        sensor_file << "\n";
    }
    sensor_file.close();
}

void CalibExRLidarImu::saveCombinedMap(string sensorName, string fileName, vector<pair<LidarFrame, SensorFrame>> aligned_sensor_buffer_)
{
    std::cout << "combined cloud begin " << sensorName << std::endl;
    Eigen::Quaterniond q_l_b_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_l_b = {0, 0, 0};

    std::cout << "current transform is " << std::endl;
    if (sensorName == "imu")
    {
        q_l_b_ = f_l_i.rot;
        t_l_b = f_l_i.tra;
    }
    else if (sensorName == "chassis")
    {
        q_l_b_ = f_l_c.rot;
        t_l_b = f_l_c.tra;
        printFrame(f_l_c);
    }

    CloudT::Ptr combined_map_(new CloudT);
    *combined_map_ += *(aligned_sensor_buffer_[0].first.cloud);

    for (int i = 1; i < aligned_sensor_buffer_.size(); i++)
    {
        auto &aligned1 = aligned_sensor_buffer_[i - 1];
        auto &aligned2 = aligned_sensor_buffer_[i];

        // calculate estimated T_l_m
        Eigen::Matrix3d R_l1_m = aligned1.first.gT.block<3, 3>(0, 0);
        Eigen::Quaterniond q_b1_w = aligned1.second.rot;
        Eigen::Quaterniond q_b2_w = aligned2.second.rot;
        Eigen::Quaterniond est_q_b2_b1 = q_b1_w.inverse() * q_b2_w;

        Eigen::Matrix3d est_R_l2_l1 = Eigen::Matrix3d(q_l_b_.inverse() * est_q_b2_b1 * q_l_b_); // 用imu变化求lidar的变化
        Eigen::Matrix3d est_R_l2_m = R_l1_m * est_R_l2_l1;
        Eigen::Matrix4d est_T_l2_m = Eigen::Matrix4d::Identity();
        est_T_l2_m.block<3, 3>(0, 0) = est_R_l2_m;
        // est_T_l2_m.block<3, 1>(0, 3) = aligned2.first.gT.block<3, 1>(0, 3);
        est_T_l2_m.block<3, 1>(0, 3) = aligned1.first.gT.block<3, 1>(0, 3) + Eigen::Matrix3d(q_l_b_) * (aligned2.second.tra - aligned1.second.tra);
        // t_l_b - est_R_l2_m * t_l_b + Eigen::Matrix3d(q_l_b_) * (aligned2.second.tra - aligned1.second.tra);

        CloudT::Ptr aligned(new CloudT);
        pcl::transformPointCloud(*aligned2.first.cloud, *aligned, est_T_l2_m);
        *combined_map_ += *aligned;
    }
    int ret = pcl::io::savePCDFileBinary("/home/xxiao/HitLidarImu/result/" + fileName + ".pcd", *combined_map_);
    std::cout << "save file---------------------------------------------------" << fileName << std::endl
              << std::endl;
}