#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  
  // State dimension
  n_x_ = 5;
  
  // Measurement dimension
  n_z_ = 3;
  
  // Augmented state dimension
  n_aug_ = 7;
  
  // Sigma dimention
  n_s_ = 2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // Weights of sigma points
  weights_ = VectorXd::Zero(n_s_);

  weights_.fill(0.5/(n_aug_ + lambda_));
  weights_(0) = lambda_/(lambda_ + n_aug_);

  // Initial state vector
  x_ = VectorXd::Zero(n_x_);

  // Initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  // Initial measurement matrix for lidar
  H_laser_ = MatrixXd::Zero(2, n_x_);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  // Initial measurement covariance matrix for lidar
  R_laser_ = MatrixXd::Zero(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;
  
  // Initial measurement covariance matrix for radar
  R_radar_ = MatrixXd::Zero(n_z_, n_z_);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;
}

UKF::~UKF() {}

// Public functions

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage &measurement_package) {

  if (!is_initialized_) {
    if (measurement_package.sensor_type_ == MeasurementPackage::RADAR) {
      const double rho     = measurement_package.raw_measurements_(0);
      const double phi     = measurement_package.raw_measurements_(1);
      const double rho_dot = measurement_package.raw_measurements_(2);
      double p_x = rho * cos(phi);
      double p_y = rho * sin(phi);
      double v_x = rho_dot * cos(phi);
      double v_y = rho_dot * sin(phi);
      double v = sqrt(v_x * v_x + v_y * v_y );
      double si = atan2(v_x,v_y);

      x_ << p_x, p_y, v, si, 0;
    }
    else if (measurement_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << measurement_package.raw_measurements_[0],
            measurement_package.raw_measurements_[1],
            0,
            0,
            0;
    }
    
    previous_timestamp_ = measurement_package.timestamp_;
    
    is_initialized_ = true;
    return;
  }
  
  // delta t - expressed in seconds
  double delta_t = (measurement_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_package.timestamp_;
  
  Predict(delta_t);
  
  if (measurement_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(measurement_package);
  }
  else if (measurement_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(measurement_package);
  }

  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Predict(double delta_t) {
  MatrixXd Xsig;
  AugmentSigmaPoints(Xsig);
  PredictSigmaPoints(delta_t, Xsig);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} measurement_package
 */
void UKF::UpdateLidar(MeasurementPackage &measurement_package) {
  UpdateLidarState(measurement_package.raw_measurements_);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} measurement_package
 */
void UKF::UpdateRadar(MeasurementPackage &measurement_package) {
  PredictRadarMeasurement();
  UpdateRadarState(measurement_package.raw_measurements_);
}


// Private functions

void UKF::AugmentSigmaPoints(MatrixXd &Xsig_out) {
  Xsig_out = MatrixXd::Zero(n_aug_, n_s_);
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  
  // Create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;
  
  /*
   Create augmented covariance matrix
   
              P_k|k  0
   P_a,k|k =
                 0   Q
   */
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
  
  /*
   Create square root matrix
   
   sqrt(P_a,k|k)
   
   */
  MatrixXd L = P_aug.llt().matrixL();
  
  /* 
   Create augmented sigma points
   
   X_a,k|k = [x_a,k|k, 
              x_a,k|k + sqrt(lambda + n_a) * sqrt(P_a,k|k), 
              x_a,k|k - sqrt(lambda + n_a) * sqrt(P_a,k|k)]
   */
  Xsig_out.col(0)  = x_aug;
  double scalar = sqrt(lambda_ + n_aug_);
  for (int i = 0; i < n_aug_; i++) {
    Xsig_out.col(i + 1)          = x_aug + scalar * L.col(i);
    Xsig_out.col(i + 1 + n_aug_) = x_aug - scalar * L.col(i);
  }
}

void UKF::PredictSigmaPoints(double delta_t, MatrixXd &Xsig) {

  Xsig_pred_ = MatrixXd::Zero(n_x_, n_s_);
  for (int i = 0; i < n_s_; i++) {
    double p_x = Xsig(0,i);
    double p_y = Xsig(1,i);
    double v = Xsig(2,i);
    const double yaw = Xsig(3,i);
    const double yawd = Xsig(4,i);
    const double nu_a = Xsig(5,i);
    const double nu_yawdd = Xsig(6,i);
    
    // Predicted state values
    double px_p, py_p;
    
    // Plug the values into the process model
    if (0.001 < fabs(yawd)) {
      px_p = p_x + v/yawd * ( sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
    }
    else { // Avoid division by zero
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }
    
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;
    
    // Add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;
    
    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;
    
    // Insert predicted sigma points into the right columns
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() {

  VectorXd x = VectorXd::Zero(n_x_);
  for (int i = 0; i < n_s_; i++) {
    x = x + weights_(i) * Xsig_pred_.col(i);
  }
  
  // Predicted state covariance matrix
  MatrixXd P = MatrixXd::Zero(n_x_, n_x_);
  for (int i = 0; i < n_s_; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    x_diff(3) = tools_.NormalizeAngle(x_diff(3));
    P = P + weights_(i) * x_diff * x_diff.transpose();
  }
  
  x_ = x;
  P_ = P;
}

void UKF::PredictRadarMeasurement() {
  /* 
   Transform sigma points into measurement space
   X_k+1|k
   ||
   \/
   z_k+1 = h(x_k+1) + w_k+1
   ||
   \/
   Z_k+1|k,i
   */
  Zsig_ = MatrixXd::Zero(n_z_, n_s_);
  for (int i = 0; i < n_s_; i++) {
    const double p_x = Xsig_pred_(0,i);
    const double p_y = Xsig_pred_(1,i);
    const double v   = Xsig_pred_(2,i);
    const double yaw = Xsig_pred_(3,i);
    const double v1 = cos(yaw) * v;
    const double v2 = sin(yaw) * v;
    
    double c = sqrt(p_x * p_x + p_y * p_y);
    Zsig_(0,i) = c;                           // rho
    Zsig_(1,i) = atan2(p_y, p_x);             // phi
    Zsig_(2,i) = (p_x * v1 + p_y * v2 ) / c;  // rho_dot
  }
  
  // Calculate predicted measurement mean
  z_pred_ = VectorXd::Zero(n_z_);
  for (int i = 0; i < n_s_; i++) {
    z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }
  
  // Calculate measurement covariance matrix
  S_ = MatrixXd::Zero(n_z_, n_z_);
  for (int i = 0; i < n_s_; i++) {
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    z_diff(1) = tools_.NormalizeAngle(z_diff(1));
    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }
  
  // Add measurement noise covariance matrix
  S_ = S_ + R_radar_;
}


void UKF::UpdateRadarState(VectorXd &z) {
  // Calculate cross correlation matrix between sigma points in state space and measurement space
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_);
  for (int i = 0; i < n_s_; i++) {
    
    // State space
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = tools_.NormalizeAngle(x_diff(3));
    
    // Measurement space
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    z_diff(1) = tools_.NormalizeAngle(z_diff(1));
    
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain and update state mean and covariance matrix
  MatrixXd K = Tc * S_.inverse();
  
  VectorXd z_diff = z - z_pred_;
  z_diff(1) = tools_.NormalizeAngle(z_diff(1));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();
}

void UKF::UpdateLidarState(VectorXd &z) {
  VectorXd y = z - H_laser_ * x_;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
  MatrixXd K = P_ * Ht * S.inverse();
  
  int size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  x_ = x_ + (K * y);
  P_ = (I - K * H_laser_) * P_;
}
