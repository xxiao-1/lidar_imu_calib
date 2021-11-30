#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <queue>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <boost/foreach.hpp>

using namespace std;
void deleteFile(string fileName)
{
  if (access(fileName.c_str(), 0) == 0)
  {
    if (remove(fileName.c_str()) == 0)
    {
      cout << fileName << " successfully deleted :)" << endl;
    }
    else
    {
      cout << "delete" << fileName << " failed :(" << endl;
    }
  }
}

Eigen::Vector3d changeTrans(Eigen::Vector3d trans, double yaw_deg, double scale)
{
  Eigen::Vector3d new_trans(0, 0, 0);
  for (int i = 0; i < 3; i++)
  {
    new_trans[i] = trans[i];
  }

  Eigen::AngleAxisd rotation_vector(yaw_deg * M_PI / 180, Eigen::Vector3d(0, 0, 1));
  Eigen::Quaterniond oriQ = Eigen::Quaterniond(rotation_vector);
  new_trans = scale * (oriQ * new_trans);
  return new_trans;
}

// deg
Eigen::Vector3d toEulerAngle(Eigen::Quaterniond q)
{
  double roll = 0, pitch = 0, yaw = 0;
  double k = 180 / M_PI;
  // roll (x-axis rotation)
  double sinr_cosp = +2.0 * (q.w() * q.x() + q.y() * q.z());
  double cosr_cosp = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
  roll = k * atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis rotation)
  double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
  if (fabs(sinp) >= 1)
    pitch = k * copysign(M_PI / 2, sinp); // use 90 degrees if out of range
  else
    pitch = k * asin(sinp);

  // yaw (z-axis rotation)
  double siny_cosp = +2.0 * (q.w() * q.z() + q.x() * q.y());
  double cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
  yaw = k * atan2(siny_cosp, cosy_cosp);
  Eigen::Vector3d res(roll, pitch, yaw);
  return res;
}

//deg
Eigen::Quaterniond yawToQuaterniond(double yaw)
{
  Eigen::AngleAxisd rotation_vector(yaw * M_PI / 180, Eigen::Vector3d(0, 0, 1));
  Eigen::Quaterniond newQ = Eigen::Quaterniond(rotation_vector);
  return newQ;
}



Eigen::Quaterniond clearRollPitch(Eigen::Quaterniond q)
{
  Eigen::Vector3d angLidar = toEulerAngle(q);
  return yawToQuaterniond(angLidar[2]);
}

#endif