#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd::Zero(estimations[0].size());
  
  int size = estimations.size();
  if (size == 0 || size != ground_truth.size()) {
    cout << "CaluculateRMSE - Error - Invalid estimation or ground_truth data" << endl;
    return rmse;
  }
  
  for (int i=0; i<size; i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  
  rmse = rmse/size;
  return rmse.array().sqrt();
}

double Tools::NormalizeAngle(double angle) {
  while (M_PI < angle) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}