// The estimator's globals, configured in memory by solution.cpp rather than a ROS parameter server.
#include "parameters.h"
#include <cmath>

double INIT_DEPTH = 5.0;
double MIN_PARALLAX = 10.0 / FOCAL_LENGTH;
// IntegrationBase multiplies these sample-level noise values by dt. The random-walk
// densities are converted so their covariance increment is density^2 * dt at 200 Hz.
double ACC_N = 0.28, ACC_W = 0.001 * std::sqrt(200.0);
double GYR_N = 0.07, GYR_W = 0.0006 * std::sqrt(200.0);
std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;
Eigen::Vector3d G{0.0, 0.0, 9.81};
double BIAS_ACC_THRESHOLD = 0.1;
double BIAS_GYR_THRESHOLD = 0.1;
double SOLVER_TIME = 1e9;
int NUM_ITERATIONS = 8;
int ESTIMATE_EXTRINSIC = 0;
int ESTIMATE_TD = 0;
int ROLLING_SHUTTER = 0;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC;
double ROW = 240, COL = 320;
double TD = 0, TR = 0;
