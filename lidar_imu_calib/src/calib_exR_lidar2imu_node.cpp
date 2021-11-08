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
#include "lidar_imu_calib/chassis_data.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace std;

queue<sensor_msgs::PointCloud2ConstPtr> lidar_buffer;
queue<sensor_msgs::ImuConstPtr> imu_buffer;
bool needLidar = false, needImu = false, needChassis = false;

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

    string data_type;
    if (!pnh.getParam("data_type", data_type))
    {
        cout << "please config param: data_type simulate or not  !!!" << endl;
        return 0;
    }
    string simulate_type;
    if (!pnh.getParam("simulate_type", simulate_type))
    {
        cout << "please config param: simulate_type double or multi  !!!" << endl;
        return 0;
    }
    vector<string> fileNames(3);
    pnh.getParam("file_name1", fileNames[0]);
    pnh.getParam("file_name2", fileNames[1]);
    pnh.getParam("file_name3", fileNames[2]);

    // initialize caliber
    CalibExRLidarImu caliber;
    caliber.sensor_buffer_1.clear();
    caliber.sensor_buffer_2.clear();
    caliber.sensor_buffer_3.clear();
    assert(caliber.sensor_buffer_1.size() == 0);
    assert(caliber.sensor_buffer_2.size() == 0);
    assert(caliber.sensor_buffer_3.size() == 0);
    // simulate data------------------------------------------------------------------
    if (data_type == "simulate")
    {
        // read new data
        for (int i = 0; i < 3; i++)
        {
            string fileName = fileNames[i];

            ifstream infile;
            infile.open(fileName);

            string sline;
            while (getline(infile, sline))
            {
                stringstream ss(sline);
                string buf;
                Frame sensor;
                Eigen::Vector4d rot;
                int j = 0;
                // cout<<ss<<endl;
                while (ss >> buf)
                {
                    if (j <= 3 && j > 0)
                    {
                        sensor.tra[j - 1] = atof(buf.c_str());
                    }
                    else if (j >= 4 && j <= 7)
                    {
                        rot[j - 4] = atof(buf.c_str());
                    }
                    j++;
                }
                sensor.rot = Eigen::Quaterniond(rot);
                if (i == 0)
                {
                    // initialize caliber
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

            std::cout << "1-2:" << std::endl;
            caliber.calibSimulateDouble(caliber.sensor_buffer_1, caliber.sensor_buffer_2, Eigen::Quaterniond(gt_M12));

            std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << "1-3:" << std::endl;
            caliber.calibSimulateDouble(caliber.sensor_buffer_1, caliber.sensor_buffer_3, Eigen::Quaterniond(gt_M13));

            std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << "2-3:" << std::endl;
            caliber.calibSimulateDouble(caliber.sensor_buffer_2, caliber.sensor_buffer_3, Eigen::Quaterniond(gt_M23));
        }
        //

        return 0;
    } // simulate

    // real data----------------------------------------------------------------------

    // read calib type
    string calib_type;
    if (!pnh.getParam("calib_type", calib_type))
    {
        cout << "please config param: calib_type !!!" << endl;
        return 0;
    }

    std::cout << "calib_type" << calib_type << endl;
    if (calib_type == "lidar2imu")
    {
        needLidar = true;
        needImu = true;
    }
    else if (calib_type == "chassis2lidar")
    {
        needLidar = true;
        needChassis = true;
    }
    else if (calib_type == "imu2chassis")
    {
        needChassis = true;
        needImu = true;
    }
    else if (calib_type == "multi")
    {
        needLidar = true;
        needChassis = true;
        needImu = true;
    }
    std::cout << "needLidar:" << needLidar << "   needImu:" << needImu << "   needChassis:" << needChassis << endl;

    // read data topic
    string lidar_topic, imu_topic, chassis_topic;
    if (!pnh.getParam("lidar_topic", lidar_topic) || !pnh.getParam("imu_topic", imu_topic) || !pnh.getParam("chassis_topic", chassis_topic))
    {
        cout << "please config param: lidar_topic, imu_topic ,chassis_topic !!!" << endl;
        return 0;
    }

    // get local param
    string bag_file;
    if (!pnh.getParam("bag_file", bag_file))
    {
        cout << "please config param: bag_file !!!" << endl;
        return 0;
    }

    // open bagfile
    rosbag::Bag bag;
    bag.open(bag_file, rosbag::bagmode::Read);
    vector<string> topics;
    topics.push_back(lidar_topic);
    topics.push_back(imu_topic);
    topics.push_back(chassis_topic);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    ros::Time imu_time;
    size_t imu_num = 0;
    double imu_delta_t;
    Eigen::Vector3d imu_velocity(0, 0, 0);
    Eigen::Vector3d imu_shift(0, 0, 0);
    Eigen::Vector3d last_imu_acc(0, 0, 0);

    ros::Time chassis_time;
    size_t chassis_num = 0;
    double chassis_delta_t;
    Eigen::Quaterniond chassis_rot = Eigen::Quaterniond::Identity();
    Eigen::Vector3d chassis_shift(0, 0, 0);
    Eigen::Vector3d last_chassis_angv(0, 0, 0);
    Eigen::Vector3d last_chassis_v(0, 0, 0);

    ofstream myfileIMU;
    bool saveIMU = true;
    ofstream myfileWheel;
    bool saveWheel = true;
    // read data and add data 逐条读取bag内消息
    foreach (rosbag::MessageInstance const m, view)
    {
        ROS_INFO_STREAM_THROTTLE(5.0, "add sensor msg ......");

        // add lidar msg
        sensor_msgs::PointCloud2ConstPtr lidar_msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (needLidar && lidar_msg != NULL)
        {
            CloudT::Ptr cloud(new CloudT);
            pcl::fromROSMsg(*lidar_msg, *cloud);
            LidarData data;
            data.cloud = cloud;
            data.stamp = lidar_msg->header.stamp.toSec();
            caliber.addLidarData(data);
        }

        // add imu msg
        sensor_msgs::ImuConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
        if (needImu && imu_msg)
        {
            imu_num++;
            ImuData data;
            data.acc = Eigen::Vector3d(imu_msg->linear_acceleration.x,
                                       imu_msg->linear_acceleration.y,
                                       imu_msg->linear_acceleration.z - 9.801);
            data.gyr = Eigen::Vector3d(imu_msg->angular_velocity.x,
                                       imu_msg->angular_velocity.y,
                                       imu_msg->angular_velocity.z);
            data.rot = Eigen::Quaterniond(imu_msg->orientation.w,
                                          imu_msg->orientation.x,
                                          imu_msg->orientation.y,
                                          imu_msg->orientation.z);
            data.stamp = imu_msg->header.stamp.toSec();
            SensorFrame SensorFrame;
            SensorFrame.stamp = data.stamp;

            if (imu_num == 1)
            {
                imu_time = imu_msg->header.stamp;
            }
            else
            {
                imu_delta_t = (imu_msg->header.stamp - imu_time).toSec();
                imu_time = imu_msg->header.stamp;
                imu_velocity = imu_velocity + (0.5 * last_imu_acc + 0.5 * data.acc) * imu_delta_t;
                imu_shift = imu_shift + imu_velocity * imu_delta_t + (0.5 * last_imu_acc + 0.5 * data.acc) * imu_delta_t * imu_delta_t / 2;
            }
            last_imu_acc = data.acc;
            SensorFrame.rot = data.rot;
            SensorFrame.tra = imu_shift;

            caliber.addImuFrame(SensorFrame);

            if (saveIMU)
            {
                myfileIMU.open("/home/xxiao/HitLidarImu/result/angvImu.txt", ios::app);
                myfileIMU.precision(10);

                myfileIMU << imu_msg->header.stamp << " ";
                myfileIMU << imu_msg->angular_velocity.x << " " << imu_msg->angular_velocity.y << " " << imu_msg->angular_velocity.z;
                myfileIMU << "\n";

                myfileIMU.close();
            }
        }

        // add chassis msg
        lidar_imu_calib::chassis_data::ConstPtr chassis_msg = m.instantiate<lidar_imu_calib::chassis_data>();
        // if (needChassis && chassis_msg)
        if (needChassis && chassis_msg)
        {
            ChassisData data;
            data = vehicleDynamicsModel(chassis_msg->header.stamp.toSec(), chassis_msg->Velocity, chassis_msg->SteeringAngle);
            if (chassis_msg->Velocity != 0 && chassis_msg->Velocity != -0)
            {
                chassis_num++;
                SensorFrame SensorFrame;
                SensorFrame.stamp = data.stamp;

                if (chassis_num == 1)
                {
                    chassis_time = chassis_msg->header.stamp;
                }
                else
                {
                    chassis_delta_t = (chassis_msg->header.stamp - chassis_time).toSec();
                    chassis_time = chassis_msg->header.stamp;

                    chassis_shift = chassis_shift + (0.5 * data.velocity + 0.5 * last_chassis_v) * chassis_delta_t;
                    // std::cout << "delta xyz" << data.velocity * chassis_delta_t << std::endl;

                    Eigen::Vector3d angle_inc = (0.5 * data.angVelocity + 0.5 * last_chassis_angv) * chassis_delta_t;
                    // std::cout << "delta angle" << angle_inc << std::endl;
                    Eigen::Quaterniond rot_inc = Eigen::Quaterniond(1.0, 0.5 * angle_inc[0], 0.5 * angle_inc[1], 0.5 * angle_inc[2]);
                    chassis_rot = chassis_rot * rot_inc;
                }
                last_chassis_v = data.velocity;
                last_chassis_angv = data.angVelocity;
                SensorFrame.rot = chassis_rot;
                SensorFrame.tra = chassis_shift;
                caliber.addChassisFrame(SensorFrame);

                if (saveWheel)
                {
                    myfileWheel.open("/home/xxiao/HitLidarImu/result/angvWheel.txt", ios::app);
                    myfileWheel.precision(10);

                    myfileWheel << chassis_msg->header.stamp << " ";
                    myfileWheel << data.angVelocity[0] << " " << data.angVelocity[1] << " " << data.angVelocity[2];
                    myfileWheel << "\n";

                    myfileWheel.close();
                }
            }
        }
    }

    // calib 结果
    if (calib_type == "lidar2imu")
    {
        caliber.calibLidar2Imu();
    }
    else if (calib_type == "chassis2lidar")
    {
        caliber.calibLidar2Chassis();
    }
    else if (calib_type == "imu2chassis")
    {
        // rpy = caliber.calibChassis2Imu();
    }
    else if (calib_type == "multi")
    {
        caliber.calibMulti();
    }

    return 0;
}