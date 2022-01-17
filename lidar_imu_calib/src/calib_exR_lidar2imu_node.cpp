#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <iostream>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <time.h>
#include "calibExRLidar2Imu.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "utils.h"
#include "lidar_imu_calib/chassis_data.h"

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace std;

queue<sensor_msgs::PointCloud2ConstPtr> lidar_buffer;
queue<sensor_msgs::ImuConstPtr> imu_buffer;
std::string chassis_origin_filename = "/home/xxiao/HitLidarImu/result/c0_chassis_origin.txt";
std::string angv_chassis_filename = "/home/xxiao/HitLidarImu/result/forKalibr/angv_chassis.txt";
std::string angv_imu_filename = "/home/xxiao/HitLidarImu/result/forKalibr/angv_imu.txt";

void lidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    lidar_buffer.push(msg);
}

void imuCallback(const sensor_msgs::ImuConstPtr &msg)
{
    imu_buffer.push(msg);
}

ChassisData vehicleDynamicsModel(double t, double Velocity, double Steer)
{
    ChassisData chassis_out;

    double steer = 0, bias = 0;
    double vx = 0, vy = 0, vz = 0, vel = 0;
    double beta;
    const double k1 = 30082 * 2; //front tyre
    const double k2 = 31888 * 2; //rear tyre
    const double mass = 1096;    //zhiliang
    const double len = 2.3;
    const double len_a = 1.0377; //qianzhou
    const double len_b = 1.2623; //houzhou
    const double i0 = 17.4;
    const double K = mass * (len_a / k2 - len_b / k1) / (len * len);

    vel = Velocity / 3.6;                 //速度
    steer = -(Steer + bias) * M_PI / 180; //方向盘转角

    beta = (1 + mass * vel * vel * len_a / (2 * len * len_b * k2)) * len_b * steer / i0 / len / (1 - K * vel * vel);
    vy = vel * sin(beta);
    vx = vel * cos(beta);
    double rz = vel * steer / i0 / len / (1 - K * vel * vel);
    chassis_out.angVelocity = {0, 0, rz};
    chassis_out.velocity = {vx, vy, vz};
    chassis_out.stamp = t;
    return chassis_out;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "calib_exR_lidar2imu_node");
    ros::NodeHandle nh, pnh("~");

    // get params
    string calib_type, lidar_topic, imu_topic, chassis_topic, bag_file, save_mode, simulate_type, data_type;
    vector<string> fileNames(3);
    pnh.getParam("file_name1", fileNames[0]);
    pnh.getParam("file_name2", fileNames[1]);
    pnh.getParam("file_name3", fileNames[2]);
    pnh.getParam("calib_type", calib_type);
    pnh.getParam("simulate_type", simulate_type);
    pnh.getParam("data_type", data_type);
    pnh.getParam("lidar_topic", lidar_topic);
    pnh.getParam("imu_topic", imu_topic);
    pnh.getParam("chassis_topic", chassis_topic);
    pnh.getParam("bag_file", bag_file);
    pnh.getParam("save_mode", save_mode);

    // initialize caliber
    CalibExRLidarImu caliber;
    caliber.sensor_buffer_1.clear();
    caliber.sensor_buffer_2.clear();
    caliber.sensor_buffer_3.clear();
    assert(caliber.sensor_buffer_1.size() == 0);
    assert(caliber.sensor_buffer_2.size() == 0);
    assert(caliber.sensor_buffer_3.size() == 0);

    // simulate part------------------------------------------------------------------
    Eigen::Vector3d time_factor(1, 1, 1);
    if (data_type == "simulate")
    {
        // read new data
        for (int i = 0; i < 3; i++)
        {
            string fileName = fileNames[i];
            Eigen::Vector3d old_tras;
            Eigen::Quaterniond old_rot;
            Eigen::Vector3d cur_tras;
            Eigen::Quaterniond cur_rot;
            bool isFirstLine = true;

            ifstream infile;
            infile.open(fileName);

            string sline;
            while (getline(infile, sline))
            {
                stringstream ss(sline);
                string buf;

                Eigen::Vector4d rot;

                // parse the line
                int j = 0;
                while (ss >> buf)
                {
                    if (j <= 3 && j > 0)
                    {
                        cur_tras[j - 1] = atof(buf.c_str());
                    }
                    else if (j >= 4 && j <= 7)
                    {
                        rot[j - 4] = atof(buf.c_str());
                    }
                    j++;
                }
                cur_rot = Eigen::Quaterniond(rot);

                if (isFirstLine)
                {
                    old_rot = cur_rot;
                    old_tras = cur_tras;
                    isFirstLine = false;
                    continue;
                }
                else
                {
                    Frame sensor;
                    sensor.rot = caliber.getInterpolatedAttitude(old_rot, cur_rot, time_factor[i]);
                    sensor.tra = caliber.getInterpolatedTranslation(old_tras, cur_tras, time_factor[i]);

                    old_rot = cur_rot;
                    old_tras = cur_tras;
                    // save
                    if (i == 0)
                    {
                        caliber.sensor_buffer_1.push_back(sensor);
                    }
                    else if (i == 1)
                    {
                        caliber.sensor_buffer_2.push_back(sensor);
                    }
                    else
                    {
                        caliber.sensor_buffer_3.push_back(sensor);
                    }
                }
            }
            infile.close();
        } // end read
        std::cout << "sensor size:" << caliber.sensor_buffer_1.size() << " " << caliber.sensor_buffer_2.size() << " " << caliber.sensor_buffer_3.size() << std::endl;
        // check read
        // cout << caliber.sensor_buffer_1.size() << "  " << caliber.sensor_buffer_2.size() << " " << caliber.sensor_buffer_3.size() << endl;
        if (simulate_type == "multi")
        {
            caliber.calibSimulateMulti(caliber.sensor_buffer_1, caliber.sensor_buffer_2, caliber.sensor_buffer_3);
        }
        else if (simulate_type == "double")
        {
            // 真值
            Eigen::Matrix3d gt_M12, gt_M13, gt_M23;
            gt_M12 << 0.82708958, -0.5569211, 0.07590595, -0.28123537, -0.52697488, -0.80200009, 0.4866513, 0.64197848, -0.59248134;
            gt_M13 << -0.55487144, 0.61344023, -0.56196866, 0.63988962, 0.74637651, 0.18292996, 0.53165681, -0.2580953, -0.80667705;
            gt_M23 = gt_M12.inverse() * gt_M13;

            Eigen::Vector3d t12(0.51410915, 0.38709595, 0.92142013);
            Eigen::Vector3d t13(0.20966865, 0.41898404, 0.0902146);
            Eigen::Vector3d t23;
            t23 = t13 - t12;

            std::cout << "1-2:" << std::endl;
            caliber.calibSimulateDouble(caliber.sensor_buffer_1, caliber.sensor_buffer_2, Eigen::Quaterniond(gt_M12), t12);

            std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << "1-3:" << std::endl;
            caliber.calibSimulateDouble(caliber.sensor_buffer_1, caliber.sensor_buffer_3, Eigen::Quaterniond(gt_M13), t13);

            std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << "2-3:" << std::endl;
            caliber.calibSimulateDouble(caliber.sensor_buffer_2, caliber.sensor_buffer_3, Eigen::Quaterniond(gt_M23), t23);
        }
        //

        return 0;
    } // simulate

    // real part----------------------------------------------------------------------

    // read bagfile
    rosbag::Bag bag;
    bag.open(bag_file, rosbag::bagmode::Read);
    vector<string> topics;
    topics.push_back(lidar_topic);
    topics.push_back(imu_topic);
    topics.push_back(chassis_topic);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    ros::Time chassis_time;
    bool first_chassis_msg = true;
    double chassis_delta_t;
    Eigen::Quaterniond chassis_rot = Eigen::Quaterniond::Identity();
    Eigen::Vector3d chassis_trans(0, 0, 0);
    Eigen::Vector3d last_chassis_angv(0, 0, 0);
    Eigen::Vector3d last_chassis_v(0, 0, 0);

    ofstream angv_imu_file;
    ofstream angv_chassis_file;
    ofstream chassis_origin_file;
    deleteFile(chassis_origin_filename);
    deleteFile(angv_chassis_filename);
    deleteFile(angv_imu_filename);

    foreach (rosbag::MessageInstance const m, view)
    {
        ROS_INFO_STREAM_THROTTLE(5.0, "add sensor msg ......");

        // add lidar msg
        sensor_msgs::PointCloud2ConstPtr lidar_msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (lidar_msg != NULL)
        {
            CloudT::Ptr cloud(new CloudT);
            pcl::fromROSMsg(*lidar_msg, *cloud);
            LidarData lidarData;
            lidarData.cloud = cloud;
            lidarData.stamp = lidar_msg->header.stamp.toSec();
            caliber.addLidarData(lidarData);
        }

        // add imu msg
        sensor_msgs::ImuConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
        if (imu_msg != NULL)
        {
            ImuData imuData;
            imuData.acc = Eigen::Vector3d(imu_msg->linear_acceleration.x,
                                          imu_msg->linear_acceleration.y,
                                          imu_msg->linear_acceleration.z - 9.801);
            imuData.gyr = Eigen::Vector3d(imu_msg->angular_velocity.x,
                                          imu_msg->angular_velocity.y,
                                          imu_msg->angular_velocity.z);
            imuData.rot = Eigen::Quaterniond(imu_msg->orientation.w,
                                             imu_msg->orientation.x,
                                             imu_msg->orientation.y,
                                             imu_msg->orientation.z);
            imuData.stamp = imu_msg->header.stamp.toSec();

            caliber.imu_raw_buffer.push_back(imuData);

            if (save_mode == "forKalibr")
            {

                angv_imu_file.open(angv_imu_filename, ios::app);
                angv_imu_file.precision(10);

                angv_imu_file << imu_msg->header.stamp << " ";
                angv_imu_file << imu_msg->angular_velocity.x << " " << imu_msg->angular_velocity.y << " " << imu_msg->angular_velocity.z;

                angv_imu_file << "\n";

                angv_imu_file.close();
            }
        }

        // add chassis msg
        lidar_imu_calib::chassis_data::ConstPtr chassis_msg = m.instantiate<lidar_imu_calib::chassis_data>();
        if (chassis_msg != NULL)
        {
            ChassisData chassisData;
            chassisData = vehicleDynamicsModel(chassis_msg->header.stamp.toSec(), chassis_msg->Velocity, chassis_msg->SteeringAngle);

            // chassis odometry
            if (chassis_msg->Velocity != 0 && chassis_msg->Velocity != -0)
            {
                SensorFrame SensorFrame;
                SensorFrame.stamp = chassisData.stamp;

                if (first_chassis_msg)
                {
                    chassis_time = chassis_msg->header.stamp;
                    first_chassis_msg = false;
                }
                else
                {
                    chassis_delta_t = (chassis_msg->header.stamp - chassis_time).toSec();
                    chassis_time = chassis_msg->header.stamp;

                    chassis_trans = chassis_trans + chassis_rot * (0.5 * chassisData.velocity + 0.5 * last_chassis_v) * chassis_delta_t;

                    Eigen::Vector3d angle_inc = (0.5 * chassisData.angVelocity + 0.5 * last_chassis_angv) * chassis_delta_t;
                    Eigen::Quaterniond rot_inc = Eigen::Quaterniond(1.0, 0.5 * angle_inc[0], 0.5 * angle_inc[1], 0.5 * angle_inc[2]);
                    chassis_rot = chassis_rot * rot_inc;
                }

                last_chassis_v = chassisData.velocity;
                last_chassis_angv = chassisData.angVelocity;

                Eigen::Vector3d pos = changeTrans(chassis_trans, 0, 0.97);
                SensorFrame.rot = chassis_rot;
                SensorFrame.tra = pos;
                caliber.addChassisFrame(SensorFrame);

                // save pos
                if (true)
                {
                    chassis_origin_file.open(chassis_origin_filename, ios::app);
                    chassis_origin_file.precision(10);

                    chassis_origin_file << chassis_msg->header.stamp << " ";
                    chassis_origin_file << pos[0] << " " << pos[1] << " " << pos[2] << " "
                                        << chassis_rot.x() << " " << chassis_rot.y() << " " << chassis_rot.z() << " " << chassis_rot.w();
                    chassis_origin_file << "\n";

                    chassis_origin_file.close();
                }

                // save angv
                if (save_mode == "forKalibr")
                {
                    angv_chassis_file.open(angv_chassis_filename, ios::app);
                    angv_chassis_file.precision(10);

                    angv_chassis_file << chassis_msg->header.stamp << " ";
                    angv_chassis_file << chassisData.angVelocity[0] << " " << chassisData.angVelocity[1] << " " << chassisData.angVelocity[2];
                    angv_chassis_file << "\n";

                    angv_chassis_file.close();
                }
            }
        }
    }

    caliber.addImuFrame(caliber.imu_raw_buffer);

    // calib 结果
    if (calib_type == "double")
    {
        std::cout << "===============================lidar 2 imu=============================================" << std::endl;
        caliber.calibLidar2Imu();
        std::cout << "===============================lidar 2 chassis=============================================" << std::endl;
        caliber.calibLidar2Chassis();
        // std::cout << "===============================imu 2 chassis=============================================" << std::endl;
        // caliber.calibImu2Chassis();
    }
    else if (calib_type == "multi")
    {
        caliber.calibMulti();
    }

    return 0;
}