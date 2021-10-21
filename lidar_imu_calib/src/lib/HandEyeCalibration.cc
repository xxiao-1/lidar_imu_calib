#include "HandEyeCalibration.h"

#include <boost/throw_exception.hpp>
#include <iostream>

#include <ceres/ceres.h>
#include "EigenUtils.h"
#include "DualQuaternion.h"

#include "ceres/ceres.h"
#include "ceres/types.h"
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <termios.h>

namespace camodocal
{

    /// @todo there may be an alignment issue, see
    /// http://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    class PoseError
    {
    public:
        PoseError(Eigen::Vector3d r1, Eigen::Vector3d t1, Eigen::Vector3d r2,
                  Eigen::Vector3d t2)
            : m_rvec1(r1), m_rvec2(r2), m_tvec1(t1), m_tvec2(t2) {}

        template <typename T>
        bool operator()(const T *const q4x1, const T *const t3x1,
                        T *residual) const
        {
            Eigen::Quaternion<T> q(q4x1[0], q4x1[1], q4x1[2], q4x1[3]);
            Eigen::Matrix<T, 3, 1> t;
            t << t3x1[0], t3x1[1], t3x1[2];

            DualQuaternion<T> dq(q, t);

            Eigen::Matrix<T, 3, 1> r1 = m_rvec1.cast<T>();
            Eigen::Matrix<T, 3, 1> t1 = m_tvec1.cast<T>();
            Eigen::Matrix<T, 3, 1> r2 = m_rvec2.cast<T>();
            Eigen::Matrix<T, 3, 1> t2 = m_tvec2.cast<T>();

            DualQuaternion<T> dq1(AngleAxisToQuaternion<T>(r1), t1);
            DualQuaternion<T> dq2(AngleAxisToQuaternion<T>(r2), t2);
            DualQuaternion<T> dq1_ = dq * dq2 * dq.inverse();

            DualQuaternion<T> diff = (dq1.inverse() * dq1_).log();
            // residual[0] = diff.real().squaredNorm() + 0.00001 * diff.dual().squaredNorm();
            residual[0] = diff.real().squaredNorm();

            return true;
        }

    private:
        Eigen::Vector3d m_rvec1, m_rvec2, m_tvec1, m_tvec2;

    public:
        /// @see
        /// http://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    class MultiError
    {
    public:
        MultiError() {}

        template <typename T>
        bool operator()(const T *const q12_4x1, const T *const t12_3x1, const T *const q31_4x1, const T *const t31_3x1, const T *const q23_4x1, const T *const t23_3x1,
                        T *residual) const
        {
            Eigen::Quaternion<T> q12(q12_4x1[0], q12_4x1[1], q12_4x1[2], q12_4x1[3]);
            Eigen::Matrix<T, 3, 1> t12;
            t12 << t12_3x1[0], t12_3x1[1], t12_3x1[2];

            DualQuaternion<T> dq12(q12, t12);

            Eigen::Quaternion<T> q31(q31_4x1[0], q31_4x1[1], q31_4x1[2], q31_4x1[3]);
            Eigen::Matrix<T, 3, 1> t31;
            t31 << t31_3x1[0], t31_3x1[1], t31_3x1[2];

            DualQuaternion<T> dq31(q31, t31);

            Eigen::Quaternion<T> q23(q23_4x1[0], q23_4x1[1], q23_4x1[2], q23_4x1[3]);
            Eigen::Matrix<T, 3, 1> t23;
            t23 << t23_3x1[0], t23_3x1[1], t23_3x1[2];

            DualQuaternion<T> dq23(q23, t23);

            DualQuaternion<T> diff = (dq12 * dq23 * dq31).log();
            residual[0] = diff.real().squaredNorm() + diff.dual().squaredNorm();

            return true;
        }

    public:
        /// @see
        /// http://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    bool HandEyeCalibration::mVerbose = true;

    HandEyeCalibration::HandEyeCalibration() {}

    void HandEyeCalibration::setVerbose(bool on) { mVerbose = on; }

    /// Reorganize data to prepare for running SVD
    /// Daniilidis 1999 Section 6, Equations (31) and (33), on page 291
    template <typename T>
    static Eigen::MatrixXd ScrewToStransposeBlockofT(
        const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &a_prime,
        const Eigen::Matrix<T, 3, 1> &b, const Eigen::Matrix<T, 3, 1> &b_prime)
    {
        Eigen::MatrixXd Stranspose(6, 8);
        Stranspose.setZero();

        typedef Eigen::Matrix<T, 3, 1> VecT;
        auto skew_a_plus_b = skew(VecT(a + b));
        auto a_minus_b = a - b;
        Stranspose.block<3, 1>(0, 0) = a_minus_b;
        Stranspose.block<3, 3>(0, 1) = skew_a_plus_b;
        Stranspose.block<3, 1>(3, 0) = a_prime - b_prime;
        Stranspose.block<3, 3>(3, 1) = skew(VecT(a_prime + b_prime));
        Stranspose.block<3, 1>(3, 4) = a_minus_b;
        Stranspose.block<3, 3>(3, 5) = skew_a_plus_b;

        return Stranspose;
    }

    /// Reorganize data to prepare for running SVD
    /// Daniilidis 1999 Section 6, Equations (31) and (33), on page 291
    // @pre no zero rotations, thus (rvec1.norm() != 0 && rvec2.norm() != 0) == true
    template <typename T>
    static Eigen::MatrixXd AxisAngleToSTransposeBlockOfT(
        const Eigen::Matrix<T, 3, 1> &rvec1, const Eigen::Matrix<T, 3, 1> &tvec1,
        const Eigen::Matrix<T, 3, 1> &rvec2, const Eigen::Matrix<T, 3, 1> &tvec2)
    {
        double theta1, d1;
        Eigen::Vector3d l1, m1;
        AngleAxisAndTranslationToScrew(rvec1, tvec1, theta1, d1, l1, m1);

        double theta2, d2;
        Eigen::Vector3d l2, m2;
        AngleAxisAndTranslationToScrew(rvec2, tvec2, theta2, d2, l2, m2);

        Eigen::Vector3d a = l1;
        Eigen::Vector3d a_prime = m1;
        Eigen::Vector3d b = l2;
        Eigen::Vector3d b_prime = m2;

        return ScrewToStransposeBlockofT(a, a_prime, b, b_prime);
    }

    void HandEyeCalibration::estimateHandEyeScrew(
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &rvecs1,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &tvecs1,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &rvecs2,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &tvecs2,
        const Eigen::Quaterniond qIn, const Eigen::MatrixXd tIn,
        Eigen::Matrix4d &H_12, bool planarMotion)
    {
        int motionCount = rvecs1.size();
        Eigen::MatrixXd T(motionCount * 6, 8);
        T.setZero();

        for (size_t i = 0; i < motionCount; ++i)
        {
            const Eigen::Vector3d &rvec1 = rvecs1.at(i);
            const Eigen::Vector3d &tvec1 = tvecs1.at(i);
            const Eigen::Vector3d &rvec2 = rvecs2.at(i);
            const Eigen::Vector3d &tvec2 = tvecs2.at(i);

            // Skip cases with zero rotation
            if (rvec1.norm() == 0 || rvec2.norm() == 0)
                continue;

            T.block<6, 8>(i * 6, 0) =
                AxisAngleToSTransposeBlockOfT(rvec1, tvec1, rvec2, tvec2);
        }

        // 获得初值
        // DualQuaternion<double> dq(qIn, tIn);
        auto dq = estimateHandEyeScrewInitial(T, planarMotion);


        mVerbose = true;
        H_12 = dq.toMatrix();
        if (mVerbose)
        {
            std::cout << "# INFO: Before refinement: H_12 = " << std::endl;
            std::cout << H_12 << std::endl;
        }

        estimateHandEyeScrewRefine(dq, rvecs1, tvecs1, rvecs2, tvecs2, qIn, tIn);

        H_12 = dq.toMatrix();
        if (mVerbose)
        {
            std::cout << "# INFO: After refinement: H_12 = " << std::endl;
            std::cout << H_12 << std::endl;
        }
    }

    // docs in header
    DualQuaterniond
    HandEyeCalibration::estimateHandEyeScrewInitial(Eigen::MatrixXd &T,
                                                    bool planarMotion)
    {

        // dq(r1, t1) = dq * dq(r2, t2) * dq.inv
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(T, Eigen::ComputeFullU |
                                                     Eigen::ComputeFullV);

        // v7 and v8 span the null space of T, v6 may also be one
        // if rank = 5.
        Eigen::Matrix<double, 8, 1> v6 = svd.matrixV().block<8, 1>(0, 5);
        Eigen::Matrix<double, 8, 1> v7 = svd.matrixV().block<8, 1>(0, 6);
        Eigen::Matrix<double, 8, 1> v8 = svd.matrixV().block<8, 1>(0, 7);

        // if rank = 5
        if (planarMotion) //(rank == 5)
        {
            if (mVerbose)
            {
                std::cout
                    << "# INFO: No unique solution, returned an arbitrary one. "
                    << std::endl;
            }

            v7 += v6;
        }

        Eigen::Vector4d u1 = v7.block<4, 1>(0, 0);
        Eigen::Vector4d v1 = v7.block<4, 1>(4, 0);
        Eigen::Vector4d u2 = v8.block<4, 1>(0, 0);
        Eigen::Vector4d v2 = v8.block<4, 1>(4, 0);

        double lambda1 = 0;
        double lambda2 = 0.0;

        if (u1.dot(v1) == 0.0)
        {
            std::swap(u1, u2);
            std::swap(v1, v2);
        }
        if (u1.dot(v1) != 0.0)
        {
            double s[2];
            solveQuadraticEquation(u1.dot(v1), u1.dot(v2) + u2.dot(v1), u2.dot(v2),
                                   s[0], s[1]);

            // find better solution for s
            double t[2];
            for (int i = 0; i < 2; ++i)
            {
                t[i] =
                    s[i] * s[i] * u1.dot(u1) + 2 * s[i] * u1.dot(u2) + u2.dot(u2);
            }

            int idx;
            if (t[0] > t[1])
            {
                idx = 0;
            }
            else
            {
                idx = 1;
            }

            double discriminant =
                4.0 * square(u1.dot(u2)) - 4.0 * (u1.dot(u1) * u2.dot(u2));
            if (discriminant == 0.0 && mVerbose)
            {
                //            std::cout << "# INFO: Noise-free case" << std::endl;
            }

            lambda2 = sqrt(1.0 / t[idx]);
            lambda1 = s[idx] * lambda2;
        }
        else
        {
            if (u1.norm() == 0 && u2.norm() > 0)
            {
                lambda1 = 0;
                lambda2 = 1.0 / u2.norm();
            }
            else if (u2.norm() == 0 && u1.norm() > 0)
            {
                lambda1 = 1.0 / u1.norm();
                lambda2 = 0;
            }
            else
            {
                std::ostringstream ss;

                ss << "camodocal::HandEyeCalibration error: normalization could "
                      "not be handled. Your rotations and translations are "
                      "probably either not aligned or not passed in properly.";
                ss << "u1:" << std::endl;
                ss << u1 << std::endl;
                ss << "v1:" << std::endl;
                ss << v1 << std::endl;
                ss << "u2:" << std::endl;
                ss << u2 << std::endl;
                ss << "v2:" << std::endl;
                ss << v2 << std::endl;
                ss << "Not handled yet. Your rotations and translations are "
                      "probably either not aligned or not passed in properly."
                   << std::endl;

                BOOST_THROW_EXCEPTION(std::runtime_error(ss.str()));
            }
        }

        // rotation
        Eigen::Vector4d q_coeffs = lambda1 * u1 + lambda2 * u2;
        Eigen::Vector4d q_prime_coeffs = lambda1 * v1 + lambda2 * v2;

        Eigen::Quaterniond q(q_coeffs(0), q_coeffs(1), q_coeffs(2), q_coeffs(3));
        Eigen::Quaterniond d(q_prime_coeffs(0), q_prime_coeffs(1),
                             q_prime_coeffs(2), q_prime_coeffs(3));

        return DualQuaterniond(q, d);
    }

    // docs in header
    bool HandEyeCalibration::solveQuadraticEquation(double a, double b, double c,
                                                    double &x1, double &x2)
    {
        double delta2 = b * b - 4.0 * a * c;

        if (delta2 < 0.0)
        {
            return false;
        }

        double delta = sqrt(delta2);

        x1 = (-b + delta) / (2.0 * a);
        x2 = (-b - delta) / (2.0 * a);

        return true;
    }

    // ceres优化
    void HandEyeCalibration::estimateHandEyeScrewRefine(
        DualQuaterniond &dq,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &rvecs1,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &tvecs1,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &rvecs2,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &tvecs2,
        const Eigen::Quaterniond qIn, const Eigen::MatrixXd tIn)
    {
        Eigen::Matrix4d H = dq.toMatrix();
        double p[7] = {dq.real().w(), dq.real().x(), dq.real().y(), dq.real().z(),
                       H(0, 3), H(1, 3), H(2, 3)};
        // std::cout << "H" << H << std::endl;
        // double p[7] = {qIn.w(), qIn.x(), qIn.y(), qIn.z(),
        //                tIn(0, 0), tIn(1, 0), tIn(2, 0)};
        ceres::Solver::Summary summary;

        ceres::Problem problem;
        for (size_t i = 0; i < rvecs1.size(); i++)
        {
            // ceres deletes the objects allocated here for the user
            ceres::CostFunction *costFunction =
                new ceres::AutoDiffCostFunction<PoseError, 1, 4, 3>(
                    new PoseError(rvecs1[i], tvecs1[i], rvecs2[i], tvecs2[i]));

            problem.AddResidualBlock(costFunction, NULL, p, p + 4); //p+4 平移量
        }

        // ceres deletes the object allocated here for the user
        ceres::LocalParameterization *quaternionParameterization =
            new ceres::QuaternionParameterization;

        problem.SetParameterization(p, quaternionParameterization);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.jacobi_scaling = true;
        options.max_num_iterations = 500;

        // ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (mVerbose)
        {
            std::cout << summary.BriefReport() << std::endl;
        }

        Eigen::Quaterniond q(p[0], p[1], p[2], p[3]);
        Eigen::Quaterniond qn = q.normalized();
        Eigen::Vector3d t;
        t << p[4], p[5], p[6];
        dq = DualQuaterniond(qn, t);
    }

    Eigen::Affine3d HandEyeCalibration::solveCeres(const EigenAffineVector &t1, const EigenAffineVector &t2, const Eigen::Quaterniond q, const Eigen::MatrixXd t)
    {

        // camera 2 ee = t2 2 t1
        using namespace camodocal;

        auto t1_it = t1.begin();
        auto t2_it = t2.begin();

        Eigen::Affine3d firstEEInverse, firstCamInverse;
        eigenVector tvecsArm, rvecsArm, tvecsFiducial, rvecsFiducial;

        bool firstTransform = true;

        for (int i = 0; i < t1.size(); ++i, ++t1_it, ++t2_it)
        {
            auto &eigenEE = *t1_it;
            auto &eigenCam = *t2_it;
            if (firstTransform)
            {
                // firstEEInverse = eigenEE.inverse();
                // firstCamInverse = eigenCam.inverse();
                firstEEInverse = Eigen::Affine3d::Identity();
                firstCamInverse = Eigen::Affine3d::Identity();
                ROS_INFO("Adding first transformation.");
                firstTransform = false;
            }
            else
            {
                Eigen::Affine3d robotTipinFirstTipBase = firstEEInverse * eigenEE; // why 除以最初的变换矩阵
                Eigen::Affine3d fiducialInFirstFiducialBase =
                    firstCamInverse * eigenCam;

                rvecsArm.push_back(eigenRotToEigenVector3dAngleAxis(
                    robotTipinFirstTipBase.rotation()));
                tvecsArm.push_back(robotTipinFirstTipBase.translation());

                rvecsFiducial.push_back(eigenRotToEigenVector3dAngleAxis(
                    fiducialInFirstFiducialBase.rotation()));
                tvecsFiducial.push_back(fiducialInFirstFiducialBase.translation());
                // ROS_INFO("Hand Eye Calibration Transform Pair Added");

                Eigen::Vector4d r_tmp = robotTipinFirstTipBase.matrix().col(3);
                r_tmp[3] = 0;
                Eigen::Vector4d c_tmp = fiducialInFirstFiducialBase.matrix().col(3);
                c_tmp[3] = 0;

                // std::cerr
                //     << "L2Norm EE: "
                //     << robotTipinFirstTipBase.matrix().block(0, 3, 3, 1).norm()
                //     << " vs Cam:"
                //     << fiducialInFirstFiducialBase.matrix().block(0, 3, 3, 1).norm()
                //     << std::endl;
            }
            // std::cerr << "EE transform: \n"
            //           << eigenEE.matrix() << std::endl;
            // std::cerr << "Cam transform: \n"
            //           << eigenCam.matrix() << std::endl;
        }

        camodocal::HandEyeCalibration calib;
        Eigen::Matrix4d result;
        calib.estimateHandEyeScrew(rvecsArm, tvecsArm, rvecsFiducial, tvecsFiducial, q, t,
                                   result, false);

        Eigen::Transform<double, 3, Eigen::Affine> resultAffine(result);

        // std::cerr << "Result from "
        //           << "EETFname"
        //           << " to "
        //           << "ARTagTFname"
        //           << ":\n"
        //           << result << std::endl;

        // std::cerr << "Translation (x,y,z) : "
        //           << resultAffine.translation().transpose() << std::endl;
        // Eigen::Quaternion<double> quaternionResult(resultAffine.rotation());
        // std::stringstream ss;
        // ss << quaternionResult.w() << ", " << quaternionResult.x() << ", "
        //    << quaternionResult.y() << ", " << quaternionResult.z() << std::endl;
        // std::cerr << "Rotation (w,x,y,z): " << ss.str() << std::endl;

        // std::cerr << "Result from "
        //           << "ARTagTFname"
        //           << " to "
        //           << "EETFname"
        //           << ":\n"
        //           << result << std::endl;
        // Eigen::Transform<double, 3, Eigen::Affine> resultAffineInv =
        //     resultAffine.inverse();
        // std::cerr << "Inverted translation (x,y,z) : "
        //           << resultAffineInv.translation().transpose() << std::endl;
        // quaternionResult = Eigen::Quaternion<double>(resultAffineInv.rotation());
        // ss.clear();
        // ss << quaternionResult.w() << " " << quaternionResult.x() << " "
        //    << quaternionResult.y() << " " << quaternionResult.z() << std::endl;
        // std::cerr << "Inverted rotation (w,x,y,z): " << ss.str() << std::endl;
        return resultAffine;
    }

    std::vector<Eigen::Affine3d> HandEyeCalibration::solveMultiCeres(
        const EigenAffineVector &t1, const EigenAffineVector &t2, const EigenAffineVector &t3,
        const Eigen::Affine3d a1, const Eigen::Affine3d a2, const Eigen::Affine3d a3)
    {

        auto t1_it = t1.begin();
        auto t2_it = t2.begin();
        auto t3_it = t3.begin();

        eigenVector tvecs1, rvecs1, tvecs2, rvecs2, tvecs3, rvecs3;

        for (int i = 0; i < t1.size(); ++i, ++t1_it, ++t2_it)
        {
            auto &eigenLidar = *t1_it;
            auto &eigenImu = *t2_it;
            auto &eigenChassis = *t3_it;

            Eigen::Affine3d deltaLidar = eigenLidar;
            Eigen::Affine3d deltaImu = eigenImu;
            Eigen::Affine3d deltaChassis = eigenChassis;

            rvecs1.push_back(eigenRotToEigenVector3dAngleAxis(
                deltaLidar.rotation()));
            tvecs1.push_back(deltaLidar.translation());

            rvecs2.push_back(eigenRotToEigenVector3dAngleAxis(
                deltaImu.rotation()));
            tvecs2.push_back(deltaImu.translation());

            rvecs3.push_back(eigenRotToEigenVector3dAngleAxis(
                deltaChassis.rotation()));
            tvecs3.push_back(deltaChassis.translation());

            // ROS_INFO("lidar imu chassis Calibration Transform Pairs Added");

            // std::cerr
            //     << "L2Norm Lidar: "
            //     << deltaLidar.matrix().block(0, 3, 3, 1).norm()
            //     << " vs Imu:"
            //     << deltaImu.matrix().block(0, 3, 3, 1).norm()
            //     << " vs Chassis:"
            //     << deltaChassis.matrix().block(0, 3, 3, 1).norm()
            //     << std::endl;

            // std::cerr << "Lidar transform: \n"
            //           << eigenLidar.matrix() << std::endl;
            // std::cerr << "Imu transform: \n"
            //           << eigenImu.matrix() << std::endl;
            // std::cerr << "Chassis transform: \n"
            //           << eigenChassis.matrix() << std::endl;
        }

        camodocal::HandEyeCalibration calib;
        Eigen::Matrix4d result1, result2, result3;

        calib.estimateMultiHandEyeScrew(rvecs1, tvecs2, rvecs1, tvecs2, rvecs1, tvecs2, a1, a2, a3,
                                        result1, result2, result3, false);

        std::cerr << "Result from "
                  << "lidar"
                  << " to "
                  << "imu"
                  << ":\n"
                  << result1 << std::endl;
        printResult(result1);

        std::cerr << "Result from "
                  << "chassis"
                  << " to "
                  << "lidar"
                  << ":\n"
                  << result2 << std::endl;
        printResult(result2);

        std::cerr << "Result from "
                  << "imu"
                  << " to "
                  << "chassis"
                  << ":\n"
                  << result3 << std::endl;
        printResult(result3);

        std::vector<Eigen::Affine3d> resultList;
        Eigen::Affine3d resultAffine1(result1);
        Eigen::Affine3d resultAffine2(result2);
        Eigen::Affine3d resultAffine3(result3);
        resultList.push_back(resultAffine1);
        resultList.push_back(resultAffine2);
        resultList.push_back(resultAffine3);
        return resultList;
    }

    void HandEyeCalibration::printResult(Eigen::Matrix4d result)
    {
        Eigen::Transform<double, 3, Eigen::Affine> resultAffine(result);
        std::cerr << "Translation (x,y,z) : "
                  << resultAffine.translation().transpose() << std::endl;
        Eigen::Quaternion<double> quaternionResult(resultAffine.rotation());
        std::stringstream ss;
        ss << quaternionResult.w() << ", " << quaternionResult.x() << ", "
           << quaternionResult.y() << ", " << quaternionResult.z() << std::endl;
        std::cerr << "Rotation (w,x,y,z): " << ss.str() << std::endl;
    }

    void HandEyeCalibration::estimateMultiHandEyeScrew(
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &rvecs1,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &tvecs1,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &rvecs2,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &tvecs2,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &rvecs3,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &tvecs3,
        const Eigen::Affine3d a12, const Eigen::Affine3d a31, const Eigen::Affine3d a23,
        Eigen::Matrix4d &H_12, Eigen::Matrix4d &H_31, Eigen::Matrix4d &H_23, bool planarMotion)
    {
        DualQuaternion<double> dq12(Eigen::Quaternion<double>(a12.rotation()), Eigen::Matrix<double, 3, 1>(a12.translation()));
        DualQuaternion<double> dq31(Eigen::Quaternion<double>(a31.rotation()), Eigen::Matrix<double, 3, 1>(a31.translation()));
        DualQuaternion<double> dq23(Eigen::Quaternion<double>(a23.rotation()), Eigen::Matrix<double, 3, 1>(a23.translation()));

        H_12 = dq12.toMatrix();
        H_31 = dq31.toMatrix();
        H_23 = dq23.toMatrix();
        std::cout << "# INFO: Before refinement: H_12 = " << std::endl;
        std::cout << H_12 << std::endl;
        std::cout << "# INFO: Before refinement: H_31 = " << std::endl;
        std::cout << H_31 << std::endl;
        std::cout << "# INFO: Before refinement: H_23 = " << std::endl;
        std::cout << H_23 << std::endl;

        estimateMultiHandEyeScrewRefine(rvecs1, tvecs1, rvecs2, tvecs2, rvecs3, tvecs3, dq12, dq31, dq23);

        H_12 = dq12.toMatrix();
        H_31 = dq31.toMatrix();
        H_23 = dq23.toMatrix();
        if (mVerbose)
        {
            std::cout << "# INFO: After refinement: H_12 = " << std::endl;
            std::cout << H_12 << std::endl;
        }
    }

    // ceres优化
    void HandEyeCalibration::estimateMultiHandEyeScrewRefine(
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &rvecs1,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &tvecs1,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &rvecs2,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &tvecs2,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &rvecs3,
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>> &tvecs3,
        DualQuaterniond &dq12, DualQuaterniond &dq31, DualQuaterniond &dq23)
    {
        Eigen::Matrix4d H12 = dq12.toMatrix();
        double p12[7] = {dq12.real().w(), dq12.real().x(), dq12.real().y(), dq12.real().z(),
                         H12(0, 3), H12(1, 3), H12(2, 3)};

        Eigen::Matrix4d H31 = dq31.toMatrix();
        double p31[7] = {dq31.real().w(), dq31.real().x(), dq31.real().y(), dq31.real().z(),
                         H31(0, 3), H31(1, 3), H31(2, 3)};

        Eigen::Matrix4d H23 = dq23.toMatrix();
        double p23[7] = {dq23.real().w(), dq23.real().x(), dq23.real().y(), dq23.real().z(),
                         H23(0, 3), H23(1, 3), H23(2, 3)};

        ceres::Solver::Summary summary;
        ceres::Problem problem;

        int posePairSize = rvecs1.size();

        for (size_t i = 0; i < posePairSize; i++)
        {
            // ceres deletes the objects allocated here for the user
            ceres::CostFunction *costFunction =
                new ceres::AutoDiffCostFunction<PoseError, 1, 4, 3>(
                    new PoseError(rvecs1[i], tvecs1[i], rvecs2[i], tvecs2[i]));

            problem.AddResidualBlock(costFunction, NULL, p12, p12 + 4); //p+4 平移量
        }

        for (size_t i = 0; i < posePairSize; i++)
        {
            // ceres deletes the objects allocated here for the user
            ceres::CostFunction *costFunction =
                new ceres::AutoDiffCostFunction<PoseError, 1, 4, 3>(
                    new PoseError(rvecs3[i], tvecs3[i], rvecs1[i], tvecs1[i]));

            problem.AddResidualBlock(costFunction, NULL, p31, p31 + 4); //p+4 平移量
        }

        for (size_t i = 0; i < posePairSize; i++)
        {
            // ceres deletes the objects allocated here for the user
            ceres::CostFunction *costFunction =
                new ceres::AutoDiffCostFunction<PoseError, 1, 4, 3>(
                    new PoseError(rvecs2[i], tvecs2[i], rvecs3[i], tvecs3[i]));

            problem.AddResidualBlock(costFunction, NULL, p23, p23 + 4); //p+4 平移量
        }

        // 相乘为I约束
        for (size_t i = 0; i < posePairSize; i++)
        {
            // ceres deletes the objects allocated here for the user
            ceres::CostFunction *costFunction =
                new ceres::AutoDiffCostFunction<MultiError, 1, 4, 3, 4, 3, 4, 3>(
                    new MultiError());

            problem.AddResidualBlock(costFunction, NULL, p12, p12 + 4, p31, p31 + 4, p23, p23 + 4); //p+4 平移量
        }

        // ceres deletes the object allocated here for the user
        ceres::LocalParameterization *quaternionParameterization =
            new ceres::QuaternionParameterization;

        problem.SetParameterization(p12, quaternionParameterization);
        problem.SetParameterization(p31, quaternionParameterization);
        problem.SetParameterization(p23, quaternionParameterization);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.jacobi_scaling = true;
        options.max_num_iterations = 500;

        // ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (mVerbose)
        {
            std::cout << summary.BriefReport() << std::endl;
        }

        Eigen::Quaterniond q12(p12[0], p12[1], p12[2], p12[3]);
        Eigen::Vector3d t12;
        t12 << p12[4], p12[5], p12[6];
        dq12 = DualQuaterniond(q12, t12);

        Eigen::Quaterniond q31(p31[0], p31[1], p31[2], p31[3]);
        Eigen::Vector3d t31;
        t31 << p31[4], p31[5], p31[6];
        dq31 = DualQuaterniond(q31, t31);

        Eigen::Quaterniond q23(p23[0], p23[1], p23[2], p23[3]);
        Eigen::Vector3d t23;
        t23 << p23[4], p23[5], p23[6];
        dq23 = DualQuaterniond(q23, t23);
    }

}